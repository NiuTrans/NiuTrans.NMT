'''
Convert a fairseq checkpoint to a NiuTrans.NMT model.
Usage: python3 model_converter.py -i $fairseq_model -o $niutrans_nmt_model -data-type <FP32/FP16>
Example: python3 model_converter.py -i fairseq.pt -o niutensor.bin -data-type FP32
Help: python3 model_converter.py -h
Requirements: fairseq>=0.6.2
'''

from tqdm import tqdm
import torch
import argparse
import numpy as np
from struct import pack, unpack

def get_model_params(model, configs, prefix=None):
    """
    Get flattened model parameters
    Args:
        model - model parameters (dict)
    """
    flattened_params = []
    encoder_embedding = None
    decoder_embedding = None
    decoder_output_weight = None

    info_file = ''
    if prefix is not None:
        info_file += prefix
    info_file += '.info.txt'
    
    print(model['encoder.history.weight'])
    #exit(0)
    with open(info_file, 'w') as f:
        for k, v, in model.items():
            v = v.to(torch.float32)
            if 'encoder.embed_tokens.weight' in k:
                encoder_embedding = v
            elif 'decoder.embed_tokens.weight' in k:
                decoder_embedding = v
            elif 'decoder.output_projection.weight' in k:
                decoder_output_weight = v
            elif v.numel() != 1:
                if 'weight' in k and 'norm' not in k:
                    if 'in_proj' in k:
                        # split qkv weights to slices
                        dim = v.shape[0] // 3
                        flattened_params.append((v[:dim, :]).t())
                        flattened_params.append((v[dim:dim*2, :]).t())
                        flattened_params.append((v[dim*2:, :]).t())
                    else:
                        if 'history.weight' in k:
                            for i, v_i in enumerate(v):
                                print(i)
                                flattened_params.append(v_i[:i+1].t())
                        else:
                            flattened_params.append(v.t())
                else:
                    flattened_params.append(v)
                f.write('{}\t\t{}\n'.format(k, v.shape))

    flattened_params.append(encoder_embedding)

    if not configs.share_all_embeddings:
        flattened_params.append(decoder_embedding)

        if not configs.share_decoder_input_output_embed:
            flattened_params.append(decoder_output_weight)

    # print(encoder_embedding.view(-1)[:10])
    # print(decoder_embedding.view(-1)[:10])
    return flattened_params

def get_model_configs(model_config, model):
    """
    Get flattened model configurations
    Args:
        model_config - model configurations (Namespace)
        model - model keys and values (dict)
    """
    if not hasattr(model_config, 'max_relative_length'):
        model_config.max_relative_length = -1
    if not hasattr(model_config, 'eos'):
        model_config.eos = 2
    if not hasattr(model_config, 'pad'):
        model_config.pad = 1
    if not hasattr(model_config, 'unk'):
        model_config.unk = 3
    flattened_configs = [
        # booleans
        'encoder.layers.0.final_layer_norm.gamma' in model.keys(),
        'decoder.layers.0.final_layer_norm.gamma' in model.keys(),
        'encoder.layers.0.self_attn.in_proj_weight' in model.keys(),
        'encoder.layer_norm.weight' in model.keys() or 'encoder.layer_norm.gamma' in model.keys(),
        'decoder.layer_norm.weight' in model.keys() or 'decoder.layer_norm.gamma' in model.keys(),
        model_config.encoder_normalize_before,
        model_config.decoder_normalize_before,
        'encoder.history.weight' in model.keys(), # place-holder for the useEncHistory flag
        'decoder.history.weight' in model.keys(), # place-holder for the useDecHistory flag
        model_config.share_all_embeddings,
        model_config.share_decoder_input_output_embed,

        # integers
        model_config.encoder_embed_dim,
        model_config.encoder_layers,
        model_config.encoder_attention_heads,
        model_config.encoder_ffn_embed_dim,
        
        model_config.decoder_embed_dim,
        model_config.decoder_layers,
        model_config.decoder_attention_heads,
        model_config.decoder_attention_heads,
        model_config.decoder_ffn_embed_dim if 'decoder.layers.0.fc1.weight' in model.keys() else -1,
        
        model_config.max_relative_length,
        model_config.max_source_positions,
        model_config.max_target_positions,

        # configurations of token ids
        model_config.eos, 
        model_config.eos, 
        model_config.pad, 
        model_config.unk,

        # source and target vocabulary size
        model['encoder.embed_tokens.weight'].shape[0],
        model['decoder.embed_tokens.weight'].shape[0],
    ]

    assert len(flattened_configs) == 29

    return flattened_configs

def save_model(configs, params, model_path, data_type):
    """
    Save model configurations and parameters to a specified path
    Args:
        configs - model configurations (list)
        params - model parameters (list)
        model_path - path to the target model file (str)
        data_type - data type of the parameters (FP32 or FP16)
    """
    int_config_list = []
    bool_config_list = []
    for c in configs:
        if isinstance(c, bool):
            bool_config_list.append(c)
        else:
            int_config_list.append(c)
    int_configs = pack('i' * len(int_config_list), *int_config_list)
    bool_configs = pack('?' * len(bool_config_list), *bool_config_list)

    with open(model_path, 'wb') as f:

        # part 1: model configurations
        f.write(bool_configs)
        f.write(int_configs)

        # part 2: values of parameters (in FP32 or FP16)
        param_num = 0
        for p in tqdm(params):
            param_num += p.numel()
            if data_type in ['fp32', 'FP32']:
                values = pack('f' * p.numel(), *(p.contiguous().view(-1).numpy().astype(np.float32)))
                f.write(values)
            elif data_type in ['fp16', 'FP16']:
                values = pack('e' * p.numel(), *(p.contiguous().view(-1).numpy().astype(np.float16)))
                f.write(values)
        print('number of parameters:', param_num)

def main():
    parser = argparse.ArgumentParser(
        description='Tool to convert fairseq checkpoint to NiuTrans.NMT model',
    )
    parser.add_argument('-i', required=False, type=str,
                        help='Input checkpoint path.')
    parser.add_argument('-o', required=False, type=str, default='',
                        help='Output model path.')
    parser.add_argument('-data-type', type=str,
                        help='Data type of the output model, FP32 (Default) or FP16',
                        default='fp32')
    args = parser.parse_args()
    print(args)

    from glob import glob
    for ckpt in glob('./*/check*.pt'):
        #args.i = ckpt
        dirname = args.i.split('/')[-2]
        #args.o = dirname + '.' + args.data_type
        print('Converting `{}` to `{}` with {}...'.format(args.i, args.o, args.data_type))

        state = torch.load(args.i, map_location='cpu')

        if 'cfg' not in state.keys():
            assert 'args' in state.keys()
            config = state['args']
        else:
            config = state['cfg']['model']
        
        cfg = vars(config)
        with open(dirname + '.info.txt', 'w', encoding='utf8') as fo:
            fo.write('*'*75)
            fo.write('\n')
            fo.write('Parameters & Shapes:\n')
            for k,v in state['model'].items():
                fo.write('{}:\t\t{}\n'.format(k,v.shape))
            fo.write('*'*75)
            fo.write('\n')
            fo.write('Training settings:\n')
            for k,v in cfg.items():
                fo.write('{}:\t\t{}\n'.format(k,v))
       
        config_list = get_model_configs(config, state['model'])
        param_list = get_model_params(state['model'], config, dirname)
        save_model(config_list, param_list, args.o, args.data_type)
        exit()

if __name__ == '__main__':
    main()
