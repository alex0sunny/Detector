from detector.filter_trigger.StaLtaTrigger import TriggerWrapper


def construct_triggers(params_list):
    triggers_objs_dic = {}
    for params_dic in params_list:
        station = params_dic['station']
        channel = params_dic['channel']
        if station not in triggers_objs_dic:
            triggers_objs_dic[station] = {}
        if channel not in triggers_objs_dic[station]:
            triggers_objs_dic[station][channel] = []
        params = ['ind', 'trigger_type', 'use_filter', 'freqmin', 'freqmax',
                  'init_level', 'stop_level', 'sta', 'lta']
        obj = TriggerWrapper(*[params_dic[param] for param in params])
        triggers_objs_dic[station][channel].append(obj)
    return triggers_objs_dic

