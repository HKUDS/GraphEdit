"""Additional information of the models."""
from collections import namedtuple
from typing import List


ModelInfo = namedtuple("ModelInfo", ["simple_name", "link", "description"])


model_info = {}


def register_model_info(
    full_names: List[str], simple_name: str, link: str, description: str
):
    info = ModelInfo(simple_name, link, description)

    for full_name in full_names:
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    if name in model_info:
        return model_info[name]
    else:
        # To fix this, please use `register_model_info` to register your model
        return ModelInfo(
            name, "", "Register the description at graphedit/model/model_registry.py"
        )


register_model_info(
    ["gpt-4"], "ChatGPT-4", "https://openai.com/research/gpt-4", "ChatGPT-4 by OpenAI"
)
register_model_info(
    ["gpt-3.5-turbo"],
    "ChatGPT-3.5",
    "https://openai.com/blog/chatgpt",
    "ChatGPT-3.5 by OpenAI",
)
register_model_info(
    ["claude-2"],
    "Claude",
    "https://www.anthropic.com/index/claude-2",
    "Claude 2 by Anthropic",
)
register_model_info(
    ["claude-1"],
    "Claude",
    "https://www.anthropic.com/index/introducing-claude",
    "Claude by Anthropic",
)
register_model_info(
    ["claude-instant-1"],
    "Claude Instant",
    "https://www.anthropic.com/index/introducing-claude",
    "Claude Instant by Anthropic",
)
register_model_info(
    ["palm-2"],
    "PaLM 2 Chat",
    "https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023",
    "PaLM 2 for Chat (chat-bison@001) by Google",
)
register_model_info(
    ["llama-2-70b-chat", "llama-2-34b-chat", "llama-2-13b-chat", "llama-2-7b-chat"],
    "Llama 2",
    "https://ai.meta.com/llama/",
    "open foundation and fine-tuned chat models by Meta",
)
register_model_info(
    ["codellama-34b-instruct", "codellama-13b-instruct", "codellama-7b-instruct"],
    "Code Llama",
    "https://ai.meta.com/blog/code-llama-large-language-model-coding/",
    "open foundation models for code by Meta",
)
register_model_info(
    [
        "vicuna-33b",
        "vicuna-33b-v1.3",
        "vicuna-13b",
        "vicuna-13b-v1.3",
        "vicuna-7b",
        "vicuna-7b-v1.3",
    ],
    "Vicuna",
    "https://lmsys.org/blog/2023-03-30-vicuna/",
    "a chat assistant fine-tuned on user-shared conversations by LMSYS",
)
register_model_info(
    ["wizardlm-70b", "wizardlm-30b", "wizardlm-13b"],
    "WizardLM",
    "https://github.com/nlpxucan/WizardLM",
    "an instruction-following LLM using evol-instruct by Microsoft",
)
register_model_info(
    ["wizardcoder-15b-v1.0"],
    "WizardLM",
    "https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder",
    "Empowering Code Large Language Models with Evol-Instruct",
)
register_model_info(
    ["mpt-7b-chat", "mpt-30b-chat"],
    "MPT-Chat",
    "https://www.mosaicml.com/blog/mpt-30b",
    "a chatbot fine-tuned from MPT by MosaicML",
)
register_model_info(
    ["guanaco-33b", "guanaco-65b"],
    "Guanaco",
    "https://github.com/artidoro/qlora",
    "a model fine-tuned with QLoRA by UW",
)
register_model_info(
    ["gpt4all-13b-snoozy"],
    "GPT4All-Snoozy",
    "https://github.com/nomic-ai/gpt4all",
    "a finetuned LLaMA model on assistant style data by Nomic AI",
)
register_model_info(
    ["koala-13b"],
    "Koala",
    "https://bair.berkeley.edu/blog/2023/04/03/koala",
    "a dialogue model for academic research by BAIR",
)
register_model_info(
    ["RWKV-4-Raven-14B"],
    "RWKV-4-Raven",
    "https://huggingface.co/BlinkDL/rwkv-4-raven",
    "an RNN with transformer-level LLM performance",
)
register_model_info(
    ["chatglm-6b", "chatglm2-6b"],
    "ChatGLM",
    "https://chatglm.cn/blog",
    "an open bilingual dialogue language model by Tsinghua University",
)
register_model_info(
    ["alpaca-13b"],
    "Alpaca",
    "https://crfm.stanford.edu/2023/03/13/alpaca.html",
    "a model fine-tuned from LLaMA on instruction-following demonstrations by Stanford",
)
register_model_info(
    ["oasst-pythia-12b"],
    "OpenAssistant (oasst)",
    "https://open-assistant.io",
    "an Open Assistant for everyone by LAION",
)
register_model_info(
    ["oasst-sft-7-llama-30b"],
    "OpenAssistant (oasst)",
    "https://open-assistant.io",
    "an Open Assistant for everyone by LAION",
)
register_model_info(
    ["llama-7b", "llama-13b"],
    "LLaMA",
    "https://arxiv.org/abs/2302.13971",
    "open and efficient foundation language models by Meta",
)
register_model_info(
    ["open-llama-7b-v2-open-instruct", "open-llama-7b-open-instruct"],
    "Open LLaMa (Open Instruct)",
    "https://medium.com/vmware-data-ml-blog/starter-llm-for-the-enterprise-instruction-tuning-openllama-7b-d05fc3bbaccc",
    "Open LLaMa fine-tuned on instruction-following data by VMware",
)
register_model_info(
    ["dolly-v2-12b"],
    "Dolly",
    "https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm",
    "an instruction-tuned open large language model by Databricks",
)
register_model_info(
    ["stablelm-tuned-alpha-7b"],
    "StableLM",
    "https://github.com/stability-AI/stableLM",
    "Stability AI language models",
)
register_model_info(
    ["codet5p-6b"],
    "CodeT5p-6b",
    "https://huggingface.co/Salesforce/codet5p-6b",
    "Code completion model released by Salesforce",
)
register_model_info(
    ["graphedit-t5-3b", "graphedit-t5-3b-v1.0"],
    "graphedit-T5",
    "https://huggingface.co/lmsys/graphedit-t5-3b-v1.0",
    "a chat assistant fine-tuned from FLAN-T5 by LMSYS",
)
register_model_info(
    ["phoenix-inst-chat-7b"],
    "Phoenix-7B",
    "https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b",
    "a multilingual chat assistant fine-tuned from Bloomz to democratize ChatGPT across languages by CUHK(SZ)",
)
register_model_info(
    ["realm-7b-v1"],
    "ReaLM",
    "https://github.com/FreedomIntelligence/ReaLM",
    "A chatbot fine-tuned from LLaMA2 with data generated via iterative calls to UserGPT and ChatGPT by CUHK(SZ) and SRIBD.",
)
register_model_info(
    ["billa-7b-sft"],
    "BiLLa-7B-SFT",
    "https://huggingface.co/Neutralzz/BiLLa-7B-SFT",
    "an instruction-tuned bilingual LLaMA with enhanced reasoning ability by an independent researcher",
)
register_model_info(
    ["h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2"],
    "h2oGPT-GM-7b",
    "https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2",
    "an instruction-tuned OpenLLaMA with enhanced conversational ability by H2O.ai",
)
register_model_info(
    ["baize-v2-7b", "baize-v2-13b"],
    "Baize v2",
    "https://github.com/project-baize/baize-chatbot#v2",
    "A chatbot fine-tuned from LLaMA with ChatGPT self-chat data and Self-Disillation with Feedback (SDF) by UCSD and SYSU.",
)
register_model_info(
    [
        "airoboros-l2-7b-2.1",
        "airoboros-l2-13b-2.1",
        "airoboros-c34b-2.1",
        "airoboros-l2-70b-2.1",
    ],
    "airoboros",
    "https://huggingface.co/jondurbin/airoboros-l2-70b-2.1",
    "an instruction-tuned LlaMa model tuned with 100% synthetic instruction-response pairs from GPT4",
)
register_model_info(
    [
        "spicyboros-7b-2.2",
        "spicyboros-13b-2.2",
        "spicyboros-70b-2.2",
    ],
    "spicyboros",
    "https://huggingface.co/jondurbin/spicyboros-70b-2.2",
    "de-aligned versions of the airoboros models",
)
register_model_info(
    ["Robin-7b-v2", "Robin-13b-v2", "Robin-33b-v2"],
    "Robin-v2",
    "https://huggingface.co/OptimalScale/robin-7b-v2-delta",
    "A chatbot fine-tuned from LLaMA-7b, achieving competitive performance on chitchat, commonsense reasoning and instruction-following tasks, by OptimalScale, HKUST.",
)
register_model_info(
    ["manticore-13b-chat"],
    "Manticore 13B Chat",
    "https://huggingface.co/openaccess-ai-collective/manticore-13b-chat-pyg",
    "A chatbot fine-tuned from LlaMa across several CoT and chat datasets.",
)
register_model_info(
    ["redpajama-incite-7b-chat"],
    "RedPajama-INCITE-7B-Chat",
    "https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat",
    "A chatbot fine-tuned from RedPajama-INCITE-7B-Base by Together",
)
register_model_info(
    [
        "falcon-7b",
        "falcon-7b-instruct",
        "falcon-40b",
        "falcon-40b-instruct",
        "falcon-180b",
        "falcon-180b-chat",
    ],
    "Falcon",
    "https://huggingface.co/tiiuae/falcon-180B",
    "TII's flagship series of large language models",
)
register_model_info(
    ["tigerbot-7b-sft"],
    "Tigerbot",
    "https://huggingface.co/TigerResearch/tigerbot-7b-sft",
    "TigerBot is a large-scale language model (LLM) with multiple languages and tasks.",
)
register_model_info(
    ["internlm-chat-7b", "internlm-chat-7b-8k"],
    "InternLM",
    "https://huggingface.co/internlm/internlm-chat-7b",
    "InternLM is a multi-language large-scale language model (LLM), developed by SHLAB.",
)
register_model_info(
    ["Qwen-7B-Chat"],
    "Qwen",
    "https://huggingface.co/Qwen/Qwen-7B-Chat",
    "Qwen is a multi-language large-scale language model (LLM), developed by Damo Academy.",
)
register_model_info(
    ["Llama2-Chinese-13b-Chat", "LLama2-Chinese-13B"],
    "Llama2-Chinese",
    "https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat",
    "Llama2-Chinese is a multi-language large-scale language model (LLM), developed by FlagAlpha.",
)
register_model_info(
    ["Vigogne-2-7B-Instruct", "Vigogne-2-13B-Instruct"],
    "Vigogne-Instruct",
    "https://huggingface.co/bofenghuang/vigogne-2-7b-instruct",
    "Vigogne-Instruct is a French large language model (LLM) optimized for instruction-following, developed by Bofeng Huang",
)
register_model_info(
    ["Vigogne-2-7B-Chat", "Vigogne-2-13B-Chat"],
    "Vigogne-Chat",
    "https://huggingface.co/bofenghuang/vigogne-2-7b-chat",
    "Vigogne-Chat is a French large language model (LLM) optimized for instruction-following and multi-turn dialogues, developed by Bofeng Huang",
)
register_model_info(
    ["mistral-7b-instruct"],
    "Mistral",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1",
    "a large language model by Mistral AI team",
)
register_model_info(
    ["deluxe-chat-v1"],
    "DeluxeChat",
    "",
    "Deluxe Chat",
)

register_model_info(
    ["zephyr-7b-alpha"],
    "Zephyr",
    "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha",
    "a chatbot fine-tuned from Mistral by Hugging Face",
)

register_model_info(
    [
        "Xwin-LM-7B-V0.1",
        "Xwin-LM-13B-V0.1",
        "Xwin-LM-70B-V0.1",
        "Xwin-LM-7B-V0.2",
        "Xwin-LM-13B-V0.2",
    ],
    "Xwin-LM",
    "https://github.com/Xwin-LM/Xwin-LM",
    "Chat models developed by Xwin-LM team",
)

register_model_info(
    ["lemur-70b-chat"],
    "Lemur-Chat",
    "https://huggingface.co/OpenLemur/lemur-70b-chat-v1",
    "an openly accessible language model optimized for both natural language and coding capabilities ",
)

register_model_info(
    ["Mistral-7B-OpenOrca"],
    "Open-Orca",
    "https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca",
    "A fine-tune of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) using [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)",
)
