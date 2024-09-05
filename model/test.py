from model.signal_model.Retnet.RetNetPool import *
import simple_parsing


from model.signal_model.Retnet.RetNet import RetNetSlidePred, RetNetRecurrent, RetNetSignalSea
import torch

status_seq = torch.randn(3, 3000).cuda()
waveform_seq = torch.randn(3, 3000, 3).cuda()
labels = {'x': torch.randn(3, 1).cuda(), 'y': torch.randn(3, 1).cuda()}
labels = {'status': torch.randint(0, 3, (3, 3000)).cuda()}
for config_type in [#RetNetSlidePhaseConfig, 
                    #RetNetRecPhaseConfig, 
                    #RetNetRecReLUPhaseConfig, 
                    #RetNetDecayReLUConfig, 
                    #RetRecurrentNoBiasPhaseConfig, 
                    #RetNetDecayNoBiasConfig,
                    RetNetRecSignalSeaConfig,
                    RetRecDecaySignalSeaConfig]:
    args = simple_parsing.parse(config_class=config_type, args=None, add_config_path_arg=True)
    print(args)
    model = eval(args.model_type)(args)          
    model = model.cuda()
    output = model(status_seq=None, waveform_seq=waveform_seq, labels=labels)
    print(output.loss)
