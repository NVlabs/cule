import re

from torchcule_atari import AtariState

game_attr_map = {
                    'Boxing'          : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'Bowling'         : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'Carnival'        : (['reward', 'score', 'terminal'], [int, int, bool], '4K'),
                    'DoubleDunk'      : (['reward', 'score', 'terminal'], [int, int, bool], 'F6'),
                    'Enduro'          : (['reward', 'score', 'terminal'], [int, int, bool], '4K'),
                    'FishingDerby'    : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'Freeway'         : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'IceHockey'       : (['reward', 'score', 'terminal'], [int, int, bool], '4K'),
                    'JourneyEscape'   : (['reward', 'score', 'terminal'], [int, int, bool], '4K'),
                    'Kaboom'          : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'Pong'            : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),
                    'PrivateEye'      : (['reward', 'score', 'terminal'], [int, int, bool], 'F8'),
                    'Skiing'          : (['reward', 'score', 'terminal'], [int, int, bool], '2K'),

                    'AirRaid'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Alien'           : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Amidar'          : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Assault'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Asterix'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Asteroids'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Atlantis'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'BankHeist'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'BattleZone'      : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'BeamRider'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Berzerk'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Centipede'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'ChopperCommand'  : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'CrazyClimber'    : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Defender'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'DemonAttack'     : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'ElevatorAction'  : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8SC'),
                    'Frostbite'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Gopher'          : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Gravitar'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Hero'            : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Jamesbond'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'E0'),
                    'Kangaroo'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Krull'           : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'KungFuMaster'    : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'MontezumaRevenge': (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'E0'),
                    'MsPacman'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'NameThisGame'    : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Phoenix'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Pitfall'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Pooyan'          : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Riverraid'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'RoadRunner'      : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F6'),
                    'Robotank'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'FE'),
                    'Seaquest'        : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Solaris'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F6'),
                    'SpaceInvaders'   : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'TimePilot'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Tutankham'       : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'E0'),
                    'UpNDown'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),
                    'Venture'         : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'VideoPinball'    : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'WizardOfWor'     : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'YarsRevenge'     : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], '4K'),
                    'Zaxxon'          : (['reward', 'score', 'terminal', 'lives'], [int, int, bool, int], 'F8'),

                    'Breakout'        : (['reward', 'score', 'terminal', 'started', 'lives'], [int, int, bool, bool, int], '2K'),
                    'Qbert'           : (['reward', 'score', 'terminal', 'last_lives', 'lives'], [int, int, bool, int, int], '4K'),
                    'StarGunner'      : (['reward', 'score', 'terminal', 'lives', 'started'], [int, int, bool, int, bool], '4K'),
                    'Tennis'          : (['reward', 'terminal', 'points', 'score'], [int, bool, int, int], '4K'),
                }

class Deserializer(object):
    def __init__(self, _bytes):
        self.bytes = _bytes
        self.TruePattern = 0xfab1fab2
        self.FalsePattern = 0xbad1bad2

    def getInt(self):
        value = int.from_bytes(self.bytes[:4], 'little')
        self.bytes = self.bytes[4:]
        return value

    def getBool(self):
        value = self.getInt()
        if value == self.TruePattern:
            value = True
        elif value == self.FalsePattern:
            value = False
        else:
            raise ValueError('Invalid boolean value: {}'.format(hex(value)))
        return value

    def getStr(self):
        strlen = self.getInt()
        value = ''.join([chr(i) for i in self.bytes[:strlen]])
        self.bytes = self.bytes[strlen:]
        return value

    def getValues(self, typelist):
        return [getattr(self, 'get' + s.__name__.capitalize())() for s in typelist]

class State(object):
    def __init__(self, state):
        self.state = AtariState()

        state = self._initialize_from_ale(state) if hasattr(state, 'unwrapped') else state

        for k in AtariState.__dict__.keys():
            if k[0] != '_':
                setattr(self.state, k, getattr(state, k))

    def _int32(self, x):
        if x > 0xFFFFFFFF:
            raise OverflowError
        if x > 0x7FFFFFFF:
            x = int(0x100000000 - x)
            if x < 2147483648:
                return -x
            else:
                return -2147483648
        return x

    def _initialize_from_ale(self, env):
        state = AtariState()

        deserializer = Deserializer(env.unwrapped.clone_full_state())

        keys = ['left_paddle', 'right_paddle', 'frame_number', 'episode_frame_number', 'string_length', 'save_system', 'md5']
        vals = deserializer.getValues([int] * 5 + [bool, str])

        for k, v in zip(keys, vals):
            setattr(state, k, v)

        name = deserializer.getStr()
        keys = ['cycles']
        vals = deserializer.getValues([int])

        for k, v in zip(keys, vals):
            setattr(state, k, v)

        name = deserializer.getStr()
        keys = ['A', 'X', 'Y', 'SP', 'IR', 'PC', 'N', 'V', 'B', 'D', 'I', 'notZ', 'C', 'executionStatus']
        vals = deserializer.getValues([int] * 6 + [bool] * 7 + [int])

        for k, v in zip(keys, vals):
            setattr(state, k, v)

        name = deserializer.getStr()
        ramSize = deserializer.getInt()
        state.ram = [deserializer.getInt() for _ in range(ramSize)] + [0] * (256 - ramSize)

        keys = ['timer',
                'intervalShift',
                'cyclesWhenTimerSet',
                'cyclesWhenInterruptReset',
                'timerReadAfterInterrupt',
                'DDRA',
                'DDRB']
        vals = deserializer.getValues([int] * 4 + [bool] + [int] * 2)

        for k, v in zip(keys, vals):
            setattr(state, k, self._int32(v) if isinstance(v, int) else v)

        name = deserializer.getStr()
        keys = [
                'clockWhenFrameStarted',
                'clockStartDisplay',
                'clockStopDisplay',
                'clockAtLastUpdate',
                'clocksToEndOfScanLine',
                'scanlineCountForLastFrame',
                'currentScanline',
                'VSYNCFinishClock',
                'enabledObjects',
                'VSYNC',
                'VBLANK',
                'NUSIZ0',
                'NUSIZ1',
                'COLUP0',
                'COLUP1',
                'COLUPF',
                'COLUBK',
                'CTRLPF',
                'playfieldPriorityAndScore',
                'REFP0',
                'REFP1',
                'PF',
                'GRP0',
                'GRP1',
                'DGRP0',
                'DGRP1',
                'ENAM0',
                'ENAM1',
                'ENABL',
                'DENABL',
                'HMP0',
                'HMP1',
                'HMM0',
                'HMM1',
                'HMBL',
                'VDELP0',
                'VDELP1',
                'VDELBL',
                'RESMP0',
                'RESMP1',
                'collision',
                'POSP0',
                'POSP1',
                'POSM0',
                'POSM1',
                'POSBL',
                'currentGRP0',
                'currentGRP1',
                'lastHMOVEClock',
                'HMOVEBlankEnabled',
                'M0CosmicArkMotionEnabled',
                'M0CosmicArkCounter',
                'dumpEnabled',
                'dumpDisabledCycle'
                ]

        vals = deserializer.getValues([int] * 19 + [bool] * 2 + [int] * 5 + [bool] * 4 + [int] * 5 + [bool] * 5 + [int] * 9 + [bool] * 2 + [int, bool, int])

        for k, v in zip(keys, vals):
            setattr(state, k, self._int32(v) if isinstance(v, int) else v)

        assert(deserializer.getStr() == 'TIASound')
        deserializer.getValues([int] * 7)

        cart = deserializer.getStr()

        game_name = re.match(r'(\w+)NoFrameskip-v4', str(env.unwrapped.spec.id)).group(1)
        ale_keys, ale_val_types, rom_type = game_attr_map[game_name]

        if rom_type in ['F6', 'F8', 'F8SC']:
            state.bank = deserializer.getInt()
        elif rom_type == 'E0':
            assert(deserializer.getInt() == 4)
            banks = deserializer.getValues([int] * 4)
            state.bank = banks[0] + (banks[1] << 4) + (banks[2] << 8) + (banks[3] << 12)

        if rom_type in ['F8SC']:
            assert(deserializer.getInt() == 128)
            state.ram[128:] = [deserializer.getInt() for _ in range(128)]

        rng_status = deserializer.getValues([int] * 4)

        keys = ['status', 'mat1', 'mat2', 'tmat']
        vals = deserializer.getValues([int] * 3)

        ale_vals = deserializer.getValues(ale_val_types)
        for k, v in zip(ale_keys, ale_vals):
            setattr(state, k, self._int32(v) if isinstance(v, int) else v)

        if not 'lives' in ale_keys:
            state.lives = 0

        assert(len(deserializer.bytes) == 0)

        return state

    def __repr__(self):
        return '\n'.join(['{} : {}'.format(k, getattr(self.state,k)) for k in AtariState.__dict__.keys() if k[0] != '_'])
