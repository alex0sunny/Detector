from detector.action.action_pipe import execute_action, ActionType


def turn(relay_n, on_off):
    if relay_n == 2:
        actionType = ActionType.relay_B
    else:
        actionType = ActionType.relay_A
    execute_action(actionType, on_off)

