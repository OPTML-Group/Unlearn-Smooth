from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM

from torch import nn


def sam_unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    beta: float = 0.1,
    coeff: float = 1.0,
    npo_coeff: float = 1.0,
    gamma: float = 0.0,
    sam_rho: float = 0.01
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',  # Save every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'        # Disable wandb
    )

    trainer = SAMIterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        beta=beta,
        coeff=coeff,
        npo_coeff=npo_coeff,
        gamma=gamma,
        sam_rho=sam_rho
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class SAMIterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 coeff: float = 1.0,
                 npo_coeff: float = 1.0,
                 gamma: float = 0.0,
                 sam_rho: float = 0.01,
                 **kwargs):

        print("#######################################################")
        print(f"Unlearn with SAM! Rho: {sam_rho}")
        print("#######################################################")
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.coeff = coeff
        self.npo_coeff = npo_coeff
        self.gamma = gamma
        self.sam_rho = sam_rho

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        self.grads = []

        super().__init__(*args, **kwargs)

    
    def training_step(self, model: nn.Module, inputs: dict) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        self.sam_state = {}
        self.sam_state["param_list"] = [p for p in model.parameters() if p.requires_grad]
        self.sam_state["eps"] = [None] * len(self.sam_state["param_list"])

        loss_forget_1 = self.forget_step(model, inputs)
        self.after_forget_step(model, inputs)
        model.zero_grad()
        self.pre_forget_step_perturbed(model, inputs)
        loss_forget_2 = self.forget_step_perturbed(model, inputs)
        self.after_forget_step_perturbed(model, inputs)
        model.zero_grad()
        loss_retain = self.retain_step(model, inputs)
        
        self.combine_and_update()
        total_loss = loss_forget_2 + self.gamma * loss_retain
        return total_loss.detach() / self.args.gradient_accumulation_steps

    def forget_step(self, model, inputs):
        loss = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(loss)
        self.sam_state["perturb_grad"] = []
        for p in self.sam_state["param_list"]:
            if p.grad is not None:
                self.sam_state["perturb_grad"].append(p.grad.detach().clone())
        return loss

    @torch.no_grad()
    def after_forget_step(self, model, inputs):
        perturb_grads = self.sam_state["perturb_grad"]
        param_list = self.sam_state["param_list"]

        device = perturb_grads[0].device
        norm_list = []
        for g in perturb_grads:
            if g is not None:
                norm_list.append(g.to(device).norm(2))
        if len(norm_list) == 0:
            grad_norm = torch.tensor(0.0, device=self.args.device)
        else:
            grad_norm = torch.stack(norm_list).norm(2)

        for i, (p, g) in enumerate(zip(param_list, perturb_grads)):
            if g is not None:
                eps = g * (self.sam_rho / grad_norm.to(g.device))
            else:
                eps = torch.zeros_like(p.data)
            self.sam_state["eps"][i] = eps

    @torch.no_grad()
    def pre_forget_step_perturbed(self, model, inputs):
        eps_list = self.sam_state["eps"]
        for p, eps in zip(self.sam_state["param_list"], eps_list):
            p.data.add_(eps)


    def forget_step_perturbed(self, model, inputs):
        loss = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(loss)

        forget_grads = []
        for p in self.sam_state["param_list"]:
            if p.grad is not None:
                forget_grads.append(p.grad.detach().clone())
            else:
                forget_grads.append(None)
        self.sam_state["forget_grad"] = forget_grads

        return loss

    @torch.no_grad()
    def after_forget_step_perturbed(self, model, inputs):
        eps_list = self.sam_state["eps"]
        for p, eps in zip(self.sam_state["param_list"], eps_list):
            p.data.sub_(eps)


    def retain_step(self, model, inputs):
        loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(loss)

        retain_grads = []
        for p in self.sam_state["param_list"]:
            if p.grad is not None:
                retain_grads.append(p.grad.detach().clone())
            else:
                retain_grads.append(None)
        self.sam_state["retain_grad"] = retain_grads
        return loss

    @torch.no_grad()
    def combine_and_update(self):
        fg_list = self.sam_state["forget_grad"]
        rg_list = self.sam_state["retain_grad"]
        param_list = self.sam_state["param_list"]

        for p, fg, rg in zip(param_list, fg_list, rg_list):
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            f_grad = fg if fg is not None else torch.zeros_like(p.data)
            r_grad = rg if rg is not None else torch.zeros_like(p.data)
            final_grad = self.npo_coeff * f_grad + self.coeff * r_grad
            p.grad.copy_(final_grad)
       
        self.sam_state.clear()


    def _compute_forget_loss(self, model, x, return_outputs=False):
        
        x_f, _ = x

        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )

        loss_f = outputs_f.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type and 'simnpo' not in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif 'simnpo' in self.loss_type:
            neg_log_ratio = - outputs_f.logits - self.gamma
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        print(f"Forget loss: {loss}")
        return loss

    def _compute_retain_loss(self, model, x, return_outputs=False):
        _, x_r = x

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )

        loss_r = outputs_r.loss

        loss = 0
        loss += loss_r

        return loss

    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
