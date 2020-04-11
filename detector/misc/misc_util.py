

def get_expr(formula_list, triggers_dic):
    expr_list = []
    for i in range(len(formula_list)):
        expr_item = formula_list[i]
        if i % 2:
            expr_list.append(expr_item)
        else:
            expr_list.append(str(triggers_dic.get(int(expr_item), False)))
    return ' '.join(expr_list)


#print(get_expr(['2', 'and not', '3', 'and', '1'], {1: True, 2: True}))

