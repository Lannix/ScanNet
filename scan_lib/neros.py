from scan_lib import nero as n

CONSTANT_EPOH = 100
CONSTANT_EPOH_1 = 10


def go_to_HELL():
    n.save_model(n.trening_model(0, CONSTANT_EPOH))  # предобучение и сохранение
    for i in range(1,2000):
        n.save_model(n.trening_model(1, CONSTANT_EPOH_1))
        n.save_model(n.trening_model(2, CONSTANT_EPOH_1))
        n.save_model(n.trening_model(3, CONSTANT_EPOH_1))




