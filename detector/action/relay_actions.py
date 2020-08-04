from detector.action.action_pipe import execute_action, ActionType


def turn(relay_n, on_off, inverse):
    if relay_n == 2:
        actionType = ActionType.relay_B
    else:
        actionType = ActionType.relay_A
    if inverse:
        on_off = not on_off
    execute_action(actionType, on_off)

