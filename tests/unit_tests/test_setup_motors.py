"""Assembly step of ``chromapi motors setup --id N``, on a fake bus."""

from chromapi.utils import setup_motors as sm


class Bus:
    """Servos by ID, each a dict of registers; reads give what was written."""

    def __init__(self, *ids):
        self.servos = {i: {} for i in ids}

    def ping_servo(self, i):
        return i in self.servos

    def write_servo_register(self, i, reg, size, value):
        if i in self.servos:
            self.servos[i][reg] = value
            if reg == sm.REG_ID:
                self.servos[value] = self.servos.pop(i)

    def read_servo_register(self, i, reg, size):
        return self.servos.get(i, {}).get(reg)


IDS = list(range(1, 13))


def test_assembly_from_12_down_to_1():
    bus = Bus(1)  # the first new servo
    assert sm.setup_one(bus, 12, IDS) == 0 and set(bus.servos) == {12}
    bus.servos[1] = {}  # the next one added to the chain
    assert sm.setup_one(bus, 11, IDS) == 0 and set(bus.servos) == {11, 12}
    for i in range(10, 1, -1):
        bus.servos[1] = {}
        assert sm.setup_one(bus, i, IDS) == 0
    bus.servos[1] = {}
    assert sm.setup_one(bus, 1, IDS) == 0  # ID 1 last: keeps it, registers written
    assert set(bus.servos) == set(IDS) and bus.servos[1][sm.REG_MAX_ACCEL] == sm.MAX_ACCEL


def test_refused_when_the_chain_or_the_new_servo_is_missing():
    assert sm.setup_one(Bus(1, 12), 10, IDS) == 1  # servo 11 of the previous step not answering
    assert sm.setup_one(Bus(12), 11, IDS) == 1  # no new servo at ID 1
    assert sm.setup_one(Bus(1), 13, IDS) == 1  # not one of the robot's IDs


def test_already_set_up_servo_keeps_its_id():
    bus = Bus(1, 12)  # a re-run of step 12, a new servo already on the bus
    assert sm.setup_one(bus, 12, IDS) == 0 and set(bus.servos) == {1, 12} and bus.servos[1] == {}
