# -*- coding: utf-8 -*-
"""
Converted from VBASim Basic Classes
initially by Yujing Lin for Python 2.7
Update to Python 3 by Linda Pei & Barry L Nelson
Last update 8/15/2018

"""

import SimClasses


def sim_funcs_init(calendar, queues, ctstats, dtstats, resources):
    # Function to initialize SimFunctions.Python
    # Typically called before the first replication and between replications
    SimClasses.Clock = 0.0
    # Empty event calendar
    while calendar.N() > 0:
        EV = calendar.Remove()
        
    # Empty queues
    # On first call, append the CStats created by FIFOQueue
    for Q in queues:
        if Q.WIP not in ctstats:
            ctstats.append(Q.WIP)
        while Q.num_queue() > 0:
            en = Q.remove()

    # Reinitialize Resources
    # On first call, append the CStats created by FIFOQueue and Resource
    for Re in resources:
        Re.Busy = 0.0
        if Re.NumBusy not in ctstats:
            ctstats.append(Re.NumBusy)  
    
    # Clear statistics
    for CT in ctstats:
        CT.clear()
        CT.Xlast = 0.0   # added bln 
        
    for DT in dtstats:
        DT.clear()
    

def schedule(calendar, event_type, event_time, **kwargs):
    # Schedule future events of EventType to occur at time SimClasses.Clock + EventTime
    added_event = SimClasses.EventNotice()
    added_event.EventType = event_type
    added_event.EventTime = SimClasses.Clock + event_time
    added_event.kwargs = kwargs
    calendar.Schedule(added_event)

    
def scheduler_plus(calendar, event_type, event_time, TheObject):
    # Schedule future events of EventType to occur at time SimClasses.Clock + EventTime
    # and pass with the event notice TheObject
    added_event = SimClasses.EventNotice()
    added_event.EventType = event_type
    added_event.EventTime = SimClasses.Clock + event_time
    added_event.WhichObject = TheObject
    calendar.Schedule(added_event)
    
    
def clear_stats(ctstats, dtstats):
    # Clear statistics in TheDTStats and TheCTStats
    for CT in ctstats:
        CT.clear()
    for DT in dtstats:
        DT.clear()
