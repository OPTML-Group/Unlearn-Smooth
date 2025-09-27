<div align='center'>
 
# Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond

[![Venue: ICML 2025](https://img.shields.io/badge/Venue-ICML%202025-green)](https://icml.cc/virtual/2025/poster/43469)
[![preprint](https://img.shields.io/badge/arXiv-2502.05374-B31B1B)](https://arxiv.org/abs/2502.05374)
[![collection](https://img.shields.io/badge/HuggingFace-Collection-yellow)](https://huggingface.co/collections/OPTML-Group/smooth-unlearned-model-67a92bb04d402b6ca3b2fb01)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/Unlearn-Smooth?tab=MIT-1-ov-file)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Smooth)](https://github.com/OPTML-Group/Unlearn-Smooth)

</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> (a) Improved unlearning robustness by smoothness-enhanced NPO (including NPO+SAM, RS, GP, CR, or WA)
compared to vanilla NPO on WMDP. (b)~(h) The prediction loss landscape
of the original model and unlearned model on the forget set.</em>
    </td>
  </tr>
</table>

This is the official code repository for ICML 2025 paper [Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond](https://arxiv.org/abs/2502.05374).

## Abstract
The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to "relearning" the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning.

## Getting Started
* [Smoothness-enhanced unlearn on WMDP](WMDP)
* [Smoothness-enhanced unlearn on MUSE](MUSE)

## Download Models
To directly using our unlearned model, please refer to our HuggingFace Collection:
* [🤗OPTML-Group/Smooth-Unlearned-Models](https://huggingface.co/collections/OPTML-Group/smooth-unlearned-model-67a92bb04d402b6ca3b2fb01)

## Contributors
* [Chongyu Fan](https://a-f1.github.io/)
* [Jinghan Jia](https://jinghanjia.netlify.app/)

## Cite This Work
```
@article{fan2025towards,
  title={Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond},
  author={Fan, Chongyu and Jia, Jinghan and Zhang, Yihua and Ramakrishna, Anil and Hong, Mingyi and Liu, Sijia},
  journal={arXiv preprint arXiv:2502.05374},
  year={2025}
}
```