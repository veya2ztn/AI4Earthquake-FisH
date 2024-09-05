from .model_arguements import SignalModelConfig, build_config_pool

def load_model(args: SignalModelConfig):
    """
    Build the model script only when using
    """
    # if 'longformer' in args.model_type:
    #     from .signal_model.LongFormerSignal import LongformerConfig, LongPrediction, LongSignalSequence

    #     LONG_Model_Pool = {
    #         'longformer': LongformerConfig,
    #         'longformer_pred': LongPrediction,
    #         'longformer_sequence': LongSignalSequence,
    #     }

    #     config = LongformerConfig(
    #         attention_window=args.attention_window,
    #         sep_token_id=0,
    #         pad_token_id=0,
    #         bos_token_id=0,
    #         eos_token_id=0,
    #         num_attention_heads=8,
    #         vocab_size=args.vocab_size,
    #         context_length=args.max_length,
    #         hidden_size=args.hidden_size,
    #         intermediate_size=args.intermediate_size,
    #         attention_hidden_size=args.attention_hidden_size,
    #         num_hidden_layers=args.num_hidden_layers)
    #     assert args.vocab_size == 4
    #     model = LONG_Model_Pool[args.model_type](config, downstream_pool)
    # elif 'rwkv' in args.model_type:
    #     from .signal_model.RWKV_Signal import (RwkvForSignal, RwkvForPrediction,
    #                             RwkvPred_MSF_Phase, RwkvOnlyWave, RwkvforSignal_Sequence)

    #     RWKV_Model_Pool = {
    #         'rwkv': RwkvForSignal,
    #         'rwkv_pred': RwkvForPrediction,
    #         'rwkv_magloc': RwkvForPrediction,
    #         'rwkvpredmsf_phase': RwkvPred_MSF_Phase,
    #         'rwkv_wave2magloc': RwkvOnlyWave,
    #         'rwkv_sequence': RwkvforSignal_Sequence,
    #     }
        
    #     model = RWKV_Model_Pool[args.model_type](args, downstream_pool)
    #     from mltool.universal_model_util import init_weights
    #     model.apply(init_weights)
    # elif 'TSL' in args.model_type:
    #     from .signal_model.TSLmodels import TimesNetSignal,DLinearSignal,PatchTSTSignal
    #     TSL_Model_Pool = {'TSL_TimesNet': TimesNetSignal,
    #                       'TSL_DLinear':DLinearSignal,
    #                       'TSL_PatchTST':PatchTSTSignal}
    #     model = TSL_Model_Pool[args.model_type](args, downstream_pool)
    # elif 'dummy' in args.model_type:
    #     from .signal_model.DummySignal import  DummyModel
    #     model = DummyModel(args, downstream_pool)
    # elif 'ViT' in args.model_type:
    #     from .signal_model.ViT_Signal import (TransformerSignalPatch, ViTALLSignalMSFMerge, ViTALLSignalMSFPure, 
    #                                   ViTALLSignalPatchMerge, TransformerSignalMultiFeature, TransformerSignalPatchMerge,
    #                                   ViTAMSFA,ViTAMSFP,ViTAMSFPNobias)
    #     ViT_Model_Pool = {'ViT_Patch': TransformerSignalPatch,
    #                       'ViT_MSFet' :TransformerSignalMultiFeature,
    #                       'ViT_Patch_Merge' :TransformerSignalPatchMerge,
    #                       'ViT_Patch_A': ViTALLSignalPatchMerge,
    #                       'ViT_MS_A': ViTALLSignalMSFMerge, #<-- this forget remove the trainable pos_embed
    #                       'ViT_MS_P': ViTALLSignalMSFPure, #<-- this forget remove the trainable pos_embed
    #                       'ViTAMSFA': ViTAMSFA, #<-- this remove the trainable pos_embed, bug fixed
    #                       'ViTAMSFP':ViTAMSFP, #<-- this remove the trainable pos_embed, bug fixed
    #                       'ViTAMSFPNobias':ViTAMSFPNobias, #<-- this is centralized model, which f(0) = 0
    #                       }
    #     #assert args.max_length == 6000,"only allow Quake6000 mode"
    #     #assert args.warning_window == 0, "only allow Quake6000 mode"
    #     model = ViT_Model_Pool[args.model_type](args, downstream_pool)
    # elif 'FNO' in args.model_type:
    #     from .signal_model.FNONet import AFNONetSignalAll,AFNONetSignalMSFAll
    #     FNO_Model_Pool = {'AFNOALL':AFNONetSignalAll,
    #                       'AFNOMSF': AFNONetSignalMSFAll}
    #     model = FNO_Model_Pool[args.model_type](args, downstream_pool)
    # elif 'Simple' in args.model_type:
    #     from .signal_model.Simplemodel import SimpleMSFASignal
    #     SimpleMSFASignal_Pool = {'SimpleMSFA': SimpleMSFASignal}
    #     model = SimpleMSFASignal_Pool[args.model_type](args, downstream_pool)
    if 'RetNet' in args.model_type:
        from .signal_model.Retnet.RetNet import RetNetSlidePred, RetNetRecurrent, RetNetSignalSea, RetNetSignalSeaL1, RetNetDirctSea,RetNetSignalLake
        from .signal_model.Retnet.RetNetGroup import RetNetGroupSlide, RetNetGroupSea
        model = eval(args.model_type)(args)
    elif 'ViT' in args.model_type:
        from .signal_model.ViT.ViT_Signal import ViTSlidePred
        model = eval(args.model_type)(args)
    elif 'Swin' in args.model_type:
        from .signal_model.SWIN.SwinTransformer import Swin2DSlidePred
        model = eval(args.model_type)(args)
    else:
        raise NotImplementedError(f"model type {args.model_type} not implemented")
    
    return model


valid_model_config_pool = (build_config_pool('model.signal_model.Retnet.RetNetPool')|
                           build_config_pool('model.signal_model.ViT.ViTPool')|
                           build_config_pool('model.signal_model.SWIN.SwinPool'))

