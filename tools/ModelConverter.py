'''
Convert a fairseq checkpoint to a NiuTrans.NMT model.
Usage: python3 ModelConverter.py -src <fairseq_models> -tgt <niutrans_nmt_model>
Help: python3 ModelConverter.py -h
Requirements: fairseq >= 0.6.2
'''

import torch
import argparse
import numpy as np
from glob import glob
from struct import pack

parser = argparse.ArgumentParser(
    description='The model converter for NiuTrans.NMT')
parser.add_argument('-src', help='The pattern used to find fairseq checkpoints, e.g., \'checkpoint*\'',
                    type=str, default='checkpoint')
parser.add_argument('-tgt', help='The file name prefix for Niutrans.NMT models',
                    type=str, default='model')
parser.add_argument('-mode', help='Storage mode, FP32 (Default) or FP16', type=str, default='fp32')
args = parser.parse_args()
args.mode = args.mode.lower()



def get_model_parameters(m):
    '''
    get flattend transformer model parameters
    '''
    p = []
    encoder_emb = None
    decoder_emb = None
    decoder_output_w = None
    for k in m['model']:
        if 'encoder.embed_tokens.weight' in k:
            encoder_emb = m['model'][k]
        elif 'decoder.embed_tokens.weight' in k:
            decoder_emb = m['model'][k]
        elif 'decoder.embed_out' in k:
            decoder_output_w = m['model'][k]
        elif m['model'][k].numel() != 1:
            # ignore fairseq version descriptions
            if 'weight' in k:
                # weights for qkv
                if 'in_proj' in k:
                    # split qkv weights to slices
                    dim = m['model'][k].shape[0] // 3
                    p.append((m['model'][k][:dim, :]).t())
                    p.append((m['model'][k][dim:dim*2, :]).t())
                    p.append((m['model'][k][dim*2:, :]).t())
                else:
                    if 'norm' in k:
                        p.append(m['model'][k])
                    else:
                        # transpose weights for matrix multiplication
                        if 'fc' in k:
                            p.append(m['model'][k].t())
                        else:
                            p.append(m['model'][k].t())
            else:
                # bias
                p.append(m['model'][k])

    # encoder embedding weight
    p.append(encoder_emb)

    # decoder embedding weight
    if decoder_emb is not None:
        p.append(decoder_emb)
    else:
        print('Sharing all embeddings')

    # decoder output weight
    if decoder_output_w is not None:
        p.append(decoder_output_w)
    else:
        print('Sharing decoder input output embeddings')

    return p


with torch.no_grad():

    model_files = glob(args.src)

    for index, model_file in enumerate(model_files):

        print('-' * 120)
        print("source model: \'{}\' ({}/{})".format(model_file, index+1, len(model_files)))
        print("target model: \'{}\'".format(args.tgt + "." + str(index)))
        model = torch.load(model_file, map_location='cpu')

        meta_info = {
            'src_vocab_size': 0,
            'tgt_vocab_size': 0,
            'encoder_layer': model['args'].encoder_layers,
            'decoder_layer': model['args'].decoder_layers,
            'ffn_hidden_size': model['args'].encoder_ffn_embed_dim,
            'hidden_size': model['args'].decoder_input_dim,
            'emb_size': model['args'].encoder_embed_dim,
            'head_num': model['args'].encoder_attention_heads,
            'max_relative_length': model['args'].max_relative_length,
            'share_all_embeddings': model['args'].share_all_embeddings,
            'share_decoder_input_output_embed': model['args'].share_decoder_input_output_embed,
            'max_source_positions': model['args'].max_source_positions,
        }

        params = get_model_parameters(model)

        print('total params: ', len(params))
        print('total params size: ', sum([p.numel() for p in params]))

        model = model['model']
        with open(args.tgt + "." + str(index) + "." +"name.txt", "w") as name_list:
            for p in model:
                name_list.write("{}\t{}\n".format(p, model[p].shape))
                if 'embed_tokens' in p:
                    if 'encoder' in p:
                        meta_info['src_vocab_size'] = model[p].shape[0]
                    else:
                        meta_info['tgt_vocab_size'] = model[p].shape[0]

        meta_info_list = [
            meta_info['encoder_layer'],
            meta_info['decoder_layer'],
            meta_info['ffn_hidden_size'],
            meta_info['hidden_size'],
            meta_info['emb_size'],
            meta_info['src_vocab_size'],
            meta_info['tgt_vocab_size'],
            meta_info['head_num'],
            meta_info['max_relative_length'],
            meta_info['share_all_embeddings'],
            meta_info['share_decoder_input_output_embed'],
            meta_info['max_source_positions'],
        ]
        print(meta_info)
        meta_info_list = [int(p) for p in meta_info_list]
        meta_info = pack("i" * len(meta_info_list), *meta_info_list)

        with open(args.tgt + "." + str(index), 'wb') as tgt:
            # part 1: meta info
            tgt.write(meta_info)
                
            # part 2: values of parameters (in FP32 or FP16)
            for p in params:
                if args.mode == 'fp32':
                    values = pack("f" * p.numel(), *
                                (p.contiguous().view(-1).cpu().numpy()))
                    tgt.write(values)
                elif args.mode == 'fp16':
                    values = pack(
                        "e" * p.numel(), *(p.contiguous().view(-1).cpu().numpy().astype(np.float16)))
                    tgt.write(values)
