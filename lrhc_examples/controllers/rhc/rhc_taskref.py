# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs

class RhcTaskRef(RhcTaskRefs):

    def __init__(self,
                n_contacts, 
                index, 
                q_remapping, 
                dtype, 
                verbose, 
                namespace = "aliengo"):
                
        super().__init__(n_contacts=n_contacts, 
                index=index, 
                q_remapping=q_remapping, 
                dtype=dtype, 
                verbose=verbose, 
                namespace=namespace)