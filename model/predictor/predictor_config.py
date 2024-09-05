#######################################
######### Predictor Config ############
#######################################

from dataclasses import dataclass
from simple_parsing import field
from typing import Union

downstream_pool_all = {
    
    "abs_magloc":  {'distance': {'metric':'fastABS', 'channel':1, 'bias':True,'nonlinear':'silu'},  # <--- [0, infinity]
                    'magnitude':{'metric':'fastABS', 'channel':1, 'bias':True,'nonlinear':'silu'},  # <--- [0, infinity]
                    # <--- [0, 1]
                    'angle':    {'metric': 'antil', 'channel': 1, 'bias': True, 'nonlinear': 'tanh'}
                    }, 
    "lp_magloc":  {'distance'  :{'metric': 'uncertainty_loss', 'channel': 1, 'bias': True, 'nonlinear': 'none'}, 
                    'magnitude':{'metric': 'uncertainty_loss', 'channel': 1, 'bias': True, 'nonlinear': 'none'}, 
                    'angle':    {'metric': 'cosine', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                   },
    "abs_locdeep":  {'distance' :{'metric':'fastABS', 'channel':1, 'bias':True,'nonlinear':'silu'},  # <--- [0, infinity]
                     'deepth'   :{'metric':'fastABS', 'channel':1, 'bias':True,'nonlinear':'silu'},  # <--- [0, infinity]
                     'angle': {'metric': 'cosine', 'channel': 1, 'bias': True, 'nonlinear':'tanh'},
                    },
    "abs_loc":  {'distance': {'metric': 'fastABS', 'channel': 1, 'bias': True, 'nonlinear': 'silu'},
                 # <--- [0, 1]
                 'angle':    {'metric': 'antil', 'channel': 1, 'bias': True, 'nonlinear': 'tanh'}
                },

    "abs_xy":  {'x': {'metric': 'fastABS', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                'y': {'metric': 'fastABS', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                       },
    "mse_xy":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
               },
    "mse_anglexy":  {'angle_x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                     'angle_y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
               },
    "mse_xy_diting_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit','unit':82.223} }, #<--distance_all std is 82, distance<110 std is 40
                            'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit','unit':83.774} }
               },
    "mse_xy_diting110_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit','unit':40} }, #<--distance_all std is 82, distance<110 std is 40
                               'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit','unit':41} }
               },
    "mse_xy_diting_gauss":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':4.180, 'std':82.233} },
                            'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer':  {'normlizer':'gaussian' ,'mean':2.918, 'std':83.758} }
               },
    "mse_xy_diting110_gauss":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.85, 'std':40} },
                                'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.47, 'std':41} }
               },
    "mse_xy_diting110_gaussb":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.85, 'std':40} },
                                 'y': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.47, 'std':41} }
               },
    "mse_xy_diting110_gaussb_L4":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.85, 'std':40} , 'layers':4},
                                    'y': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'gaussian','mean':0.47, 'std':41} , 'layers':4}
               },
    "mse_xyL4_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':29.0693025099 } , 'layers':4},
                               'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':31.4353829701 } , 'layers':4}, ### <---- this is for the STEAD
                                 },
    "mse_xymL4S_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':29.0693025099 } , 'layers':4},
                                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':31.4353829701 } , 'layers':4}, 
                                 'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                 'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
                                 },
    "mse_xymL4SFindP_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':29.0693025099 } , 'layers':4},
                                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':31.4353829701 } , 'layers':4}, 
                                 'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                 'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'},
                                 'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                 },
    "mse_xymL4SFindPS_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':29.0693025099 } , 'layers':4},
                                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':31.4353829701 } , 'layers':4}, 
                                 'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                 'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'},
                                 'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                 'findS':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                 },
    "mse_xymSL4_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':29.0693025099 } , 'layers':4},
                                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':31.4353829701 } , 'layers':4}, 
                                 'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                 'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none' , 'layers':4}

                                 },
    "mse_xymSPS_Instance_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':35.81029928774641 } , 'layers':1},
                                    'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':32.505034550055434} , 'layers':1}, 
                                    'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                     'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none' , 'layers':1},
                                    'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                    'findS':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                 },
    "mse_xymSP_Instance_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':35.81029928774641 } , 'layers':1},
                                    'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'normlizer': {'normlizer':'unit', 'unit':32.505034550055434} , 'layers':1}, 
                                    'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                     'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none' , 'layers':1},
                                    'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                                    },

    "mse_QxQymL4S_steadBO_unit":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'absunit', 'unit':29.0693025099 } , 'layers':4},
                                   'y': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'normlizer': {'normlizer':'absunit', 'unit':31.4353829701 } , 'layers':4}, 
                                   'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
                                 },
    "mse_xy_L4":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4},
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4}, ### <---- this is for the STEAD
                                 },
    "mse_xyz":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'deepth': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
               },
    "mse_xL4":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    "mse_yL4":{'y':   {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    
    "mse_quarter_xyL4":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'abs','unit':1}},
                        'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'abs','unit':1}}},
    "mse_quarter_xySL4":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'abs','unit':1}},
                         'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'abs','unit':1}},
                         'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}
                         },
    "mse_quarter_xy_N":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':1, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                       'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':1, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}}
                       },
    "mse_quarter_xyS_N":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':1, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                       'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':1, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}},
                  'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':1}
                       },
    "mse_findsharPeakL24CNN_qtxyL4":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                                     'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}},
                                 'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "S_m_Qx_Qy_findP":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                             'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}},
                         'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                         'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                         'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}
                         },
    "S_m_Qx_Qy":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                 'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}},
                 'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                 'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}
                         },
    "S_m_Qx_Qy_findP_findS":{'x':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':56.29495758072879}},
                             'y':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4, 'normlizer':{'normlizer':'absunit','unit':57.53008639206075}},
                         'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                         'findS':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                         'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                         'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}
                         },
    
    "mse_mL4":{'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4}},
    "mse_SL4":{'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    "mse_dL4":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4}},
    "mse_mdL4":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},},
    "cos_avL4":  {'angle_vector':   {'metric': 'cosinesimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    "ce_ESWN": {'ESWN':   {'metric': 'ce', 'channel': 4, 'bias': False, 'nonlinear': 'none'}},
    "ce_SPIN": {'SPIN':   {'metric': 'ce', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "mse_dmSL4":{
        'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
        'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4},
        'distance':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4}

    },
    "mse_x":   {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' }},
    "mse_xb":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none' }},
    "mse_y":   {'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' }},
    "mse_yb":  {'y': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none' }},
    "mse_xyzm":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'deepth': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                  'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
               },
    "mse_xyzmP":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                 'deepth': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                  'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                  'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xyzmS":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'deepth': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                   'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xyzL4S":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'deepth': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xyL4S":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' , 'layers':4},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xyL4":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' , 'layers':4},
                  'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' , 'layers':4}
               },
    "mse_xL4yS":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' , 'layers':4},
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xymS":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xymL4S":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none' },
                   'magnitude': {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':4},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
               },
    "mse_xyS":  {'x': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none'},
                   'y': {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none'},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
                   },
    "mse_xy_L4":  {    'x': {'metric': 'MSE',  'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4},
                        'y': {'metric': 'MSE',  'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4}
                   },
    
    "mse_xyS_L4":  {    'x': {'metric': 'MSE',  'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4},
                        'y': {'metric': 'MSE',  'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4},
                   'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
                   },
    "status":  {'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}},
    "statusL4":  {'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    "mse_magnitude":{'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'}},

    "mse_MagS":{'magnitude':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                'status':      {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}},
    "mse_distance":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "mse_distanceb":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'}},
    "mse_distancebS":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
                      'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}},
    #"mse_distance_normed":{'distance':   {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'unit': 50}},
    "abs_angle":   {'angle':   {'metric': 'antil', 'channel': 1, 'bias': True, 'nonlinear': 'tanh'} },
    "abs_angle2":  {'angle':   {'metric': 'cosine', 'channel': 1, 'bias': True, 'nonlinear': 'tanh'}},
    "mse_angle":   {'angle':   {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "mse_line" :   {'line':    {'metric': 'MSE', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "abs_angle3":  {'angle':   {'metric':'fastABS', 'channel':1, 'bias':True,'nonlinear':'tanh'} },
    "abs_angle4":  {'angle':   {'metric': 'cosine', 'channel': 1, 'bias': True, 'nonlinear': 'none'}},
    "hasP":  {'hasP':   {'metric': 'adace', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "findP":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "findPL4":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    "findsharPeakL4":     {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':4, 'criterion_config':{'std':0.005}}},
    "findsharPeakL8CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':8, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findsharPeakL16CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':16, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findsharPeakL24CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findJudgesharPeakL24CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 
                                             'module_type': 'cnn', 'criterion_config':{'std':0.005, 'judger_alpha': 0.1}}},
    "findJudgeBsharPeakL24CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 
                                             'module_type': 'cnn', 'criterion_config':{'std':0.005, 'judger_alpha': 0.01}}},
    "ESWNL24CNN":  {'ESWN':   {'metric': 'ce', 'channel': 4, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findsharSeakL24CNN":  {'findS':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findsharPSeakL24CNN":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}},
                             'findS':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':24, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    
    "findsharPeakL8CNN2":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':8, 'module_type': 'cnn2', 'criterion_config':{'std':0.005}}},
    "findsharPeakL16CNN2":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': False, 'nonlinear': 'none', 'layers':16, 'module_type': 'cnn2', 'criterion_config':{'std':0.005}}},
    "findsharPeakL16CNNB":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':16, 'module_type': 'cnn', 'criterion_config':{'std':0.005}}},
    "findsharPeakL16CNNReLU":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':16, 'module_type': 'cnn', 'intermediate_nonlinear': 'relu', 'criterion_config':{'std':0.005}}},
    
    "findsharPeakL16ResReLU":  {'findP':   {'metric': 'dbpos', 'channel': 1, 'bias': True, 'nonlinear': 'none', 'layers':16, 'module_type': 'cnn', 'intermediate_nonlinear': 'relu', 'shortcut':True,'criterion_config':{'std':0.005}}},
    
    "prob":  {'phase_probability':       {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}},
    "peakprob":  {'probabilityPeak':     {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}},
    "probS":     {'phase_probability':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'},
                  'status':              {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
                 },
    "p_peakprob": {'P_Peak_prob':    {'metric': 'focal', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "p_peakprob_bce": {'P_Peak_prob':    {'metric': 'bce', 'channel': 1, 'bias': False, 'nonlinear': 'none'}},
    "abs_angle_vector":  {'angle_vector':   {'metric': 'cosinesimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "abs_line_vector":   {'angle_vector':   {'metric': 'parallelQsimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "cos_line_vector":   {'line_vector':    {'metric': 'cosinesimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "cos_angle_vector":  {'angle_vector':   {'metric': 'cosinesimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none'}},
    "abs_angle_vector_L4":  {'angle_vector':   {'metric': 'cosinesimilirity', 'channel': 2, 'bias': False, 'nonlinear': 'none', 'layers':4}},
    
    "mse_group_vector":  {
        'distance':       {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
        'group_vector':   {'metric': 'vector_distance', 'channel': 2, 'bias': False, 'nonlinear': 'none', 'layers':0}
        },
    "mse_group_vector_S":  {
        'distance':       {'metric': 'MSE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
        'group_vector':   {'metric': 'vector_distance', 'channel': 2, 'bias': False, 'nonlinear': 'none', 'layers':0},
        'status':   {'metric': 'ce', 'channel': 3, 'bias': False, 'nonlinear': 'none'}
        },
    "mae_group_vector":  {
        'distance':       {'metric': 'MAE', 'channel': 1, 'bias': True, 'nonlinear': 'none'},
        'group_vector':   {'metric': 'vector_distance', 'channel': 2, 'bias': False, 'nonlinear': 'none', 'layers':0}
        },
    }
    


@dataclass
class PredictorConfig:
    prediction_type: str = None
    downstream_pool: Union[dict,str] = None
    downstream_task: str = field(default=None)
    normlize_at_downstream: bool = field(default=False)
    use_confidence: str = field(default='whole_sequence')
    downstream_dropout: float = field(default=0.0)
    slide_feature_window_size: int = None
    slide_stride_in_training: int = 1
    focal_loss_alpha: float = field(default=0.25)
    focal_loss_gamma: float = field(default=2)
    def __post_init__(self):
        assert self.downstream_task is not None, "downstream_task must be assigned"
        self.downstream_pool = downstream_pool_all[self.downstream_task]
        # elif isinstance(self.downstream_pool, str):
        #     self.downstream_pool = eval(self.downstream_pool)
    # def get_downstream_pool(self):
    #     if self.downstream_pool is None:
    #         self.downstream_pool = downstream_pool_all[self.downstream_task]
    #     return self.downstream_pool

@dataclass
class SlideWindowPredictorConfig(PredictorConfig):
    """
    if use slide window than we need specify the output token
    """
    prediction_type: str = 'slide_window'
    merge_token: str = field(default="average")
    
@dataclass
class RecurrentPredictorConfig(PredictorConfig):
    """
    if use slide window than we need specify the output token
    """
    prediction_type: str = 'recurrent'