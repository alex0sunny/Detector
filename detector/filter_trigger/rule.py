from time import sleep

import zmq
from obspy import UTCDateTime
import detector.misc.globals as glob

from detector.misc.globals import Port, Subscription, logger
from detector.misc.misc_util import get_expr


def rule_picker(rule_id, triggerings, triggers_ids, formula_list):
    rules_triggerings = []
    vals_dic = {trigger_id: glob.LAST_TRIGGERINGS[trigger_id] for trigger_id in triggers_ids}
    for date_time, triggering, trigger_id in triggerings:
        if trigger_id not in triggers_ids:
            continue
        vals_dic[trigger_id] = triggering
        rule_expr = get_expr(formula_list, vals_dic)
        rule_val = int(eval(rule_expr))
        if rule_val != glob.LAST_RTRIGGERINGS[rule_id]:
            logger.info('rule triggered, rule_id:' + str(rule_id) + ' rule val:' +
                        str(rule_val) + ' rule expr:' + rule_expr)
            glob.LAST_RTRIGGERINGS[rule_id] = rule_val
            actions_triggerings.extend([(date_time, (1 if rule_val else -1), action_id)
                                        for action_id in actions_ids])
    return rules_triggerings

