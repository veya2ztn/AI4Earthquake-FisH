from .utils import findAllPS, findAllP_Peak
from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
import numpy as np

@dataclass
class StatusDeterminer:
    def __call__(self, batchdata):
        raise NotImplementedError

@dataclass
class StatusDeterminer_Threshold(StatusDeterminer):
    p_threshold: int # unit is 100, thus 80 mean prob=0.8,
    s_threshold: int # unit is 100, thus 80 mean prob=0.8,
    
    def __call__(self, batchdata):# batchdata: (L,3)
        status_longlife = np.zeros_like(batchdata[...,0])
        status_longlife[batchdata[...,1]>self.p_threshold]=1
        status_longlife[batchdata[...,2]>self.s_threshold]=2
        return status_longlife
    
@dataclass
class StatusDeterminer_Max(StatusDeterminer):
    def __call__(self, batchdata):
        status_longlife = np.argmax(batchdata, -1)
        return status_longlife


@dataclass
class PhasePickingStratagy:
    def __call__(self, batchdata):
        raise NotImplementedError

@dataclass
class PhasePickingStratagy_Status(PhasePickingStratagy):
    expansion: int
    status_strategy:StatusDeterminer=StatusDeterminer_Threshold(p_threshold=80,s_threshold=80)
    windows_size: int = 7
    judger: float = 0.98
    timetype: str = field(default='realtime', choice=['realtime','posttime'])
    def __call__(self, batchdata):
        status_longlife = self.status_strategy(batchdata)
        p_position_map_pool_dual, s_position_map_pool_dual = findAllPS(status_longlife, self.windows_size,self.expansion, judger=self.judger, timetype=self.timetype)
        if self.timetype!='all':
            return p_position_map_pool_dual[self.timetype], s_position_map_pool_dual[self.timetype]
        return p_position_map_pool_dual, s_position_map_pool_dual
    
@dataclass
class PhasePickingStratagy_Prob(PhasePickingStratagy):
    tri_th_l: int # unit is 100, thus 80 mean prob=0.8,# unit is 100, thus 80 mean prob=0.8,
    tri_th_r: int # unit is 100, thus 80 mean prob=0.8,
    expansion: int
    output_one: bool
    offset: int
    end: int = None
    def __call__(self, batchdata):
        end = batchdata.shape[1] if self.end is None else self.end 
        p_position_map_pool = findAllP_Peak(batchdata[:,self.offset:end,1],self.tri_th_l,self.tri_th_r,self.expansion,output_one=self.output_one,offset=self.offset*self.expansion)
        s_position_map_pool = findAllP_Peak(batchdata[:,self.offset:end,2],self.tri_th_l,self.tri_th_r,self.expansion,output_one=self.output_one,offset=self.offset*self.expansion)   
        return p_position_map_pool, s_position_map_pool