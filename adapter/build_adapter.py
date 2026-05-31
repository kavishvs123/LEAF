from .basic_adapter import BasicAdapter
from .aug_adapter import AugAdapter
from .selectors import *
from cfg import args
import os


def build_adapter_from_cfg():
    if args.selector_type == 'optimal':
        selector = OptimalSelector(os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices{args.postfix}.json'))
    elif args.selector_type == 'llm':
        selector = LLMSelectorFromJson(
            # [CHANGED] use args.llm_postfix (not args.postfix) so --llm_postfix _lora/_qlora
            # selects the right LLM output file without renaming. The choices dump files
            # below still use args.postfix as they are shared across all model variants.
            f'./data/llm_output/{args.dataset.lower()}_output{args.llm_postfix}.json',
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices{args.postfix}.json'),
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r2{args.postfix}.json')
        )
    elif args.selector_type == 'llm_r2':
        selector = LLMSelectorFromJson(
            # [CHANGED] same as above — llm_postfix for the LLM output, postfix for choices
            f'./data/llm_output/{args.dataset.lower()}_output_r2{args.llm_postfix}.json',
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r2{args.postfix}.json'),
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r3{args.postfix}.json'),
        )
    else:
        raise ValueError('Selector not supported')
    
    if args.adapter_type == 'basic':
        return BasicAdapter(selector)
    elif args.adapter_type == 'aug':
        return AugAdapter(selector)
    else:
        raise ValueError('Adapter not supported')
