

def get_expr(formula_list, triggers_dic):
    expr_list = []
    for i in range(len(formula_list)):
        expr_item = formula_list[i]
        if i % 2:
            expr_list.append(expr_item)
        else:
            expr_list.append(str(triggers_dic.get(int(expr_item), False)))
    return ' '.join(expr_list)


def get_formula_triggers(formula_list):
    triggers = []
    for i in range(len(formula_list)):
        if not i % 2:
            trigger_id = int(formula_list[i])
            triggers.append(trigger_id)
    return triggers


#print(get_formula_triggers(['1', 'or', '2', 'and', '3']))
#print(get_expr(['2', 'and not', '3', 'and', '1'], {1: True, 2: True}))

