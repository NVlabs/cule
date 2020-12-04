#include <cule/macros.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/state.hpp>

#include <torchcule/atari_env.hpp>

struct encode_states_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const cule::atari::rom& cart,
                    const AtariState* src_states,
                    cule::atari::state* dst_states,
                    cule::atari::frame_state*,
                    uint8_t* input_ram) const
    {
        using Environment_t = cule::atari::environment<cule::atari::ROM_2K>;
        using ALE_t = typename Environment_t::ALE_t;
        using TIA_t = typename Environment_t::TIA_t;

        const size_t index = self.index();

        const AtariState& ts = src_states[index];
        cule::atari::state& s = dst_states[index];
        /* cule::atari::frame_state& fs = dst_frame_states[index]; */

        s.resistance = ts.left_paddle;
        Environment_t::setFrameNumber(s, cule::atari::ENV_BASE_FRAMES + 10 + ts.frame_number);

        s.cpuCycles = ts.cycles;

        s.A = ts.A;
        s.X = ts.X;
        s.Y = ts.Y;
        s.SP = ts.SP;
        s.PC = ts.PC;
        UPDATE_FIELD(s.sysFlags.asBitField(), cule::atari::FIELD_SYS_INT, ts.IR);

        s.sysFlags.template change<cule::atari::FLAG_NEGATIVE>(ts.N);
        s.sysFlags.template change<cule::atari::FLAG_OVERFLOW>(ts.V);
        s.sysFlags.template change<cule::atari::FLAG_BREAK>(ts.B);
        s.sysFlags.template change<cule::atari::FLAG_DECIMAL>(ts.D);
        s.sysFlags.template change<cule::atari::FLAG_INTERRUPT_OFF>(ts.I);
        s.sysFlags.template change<cule::atari::FLAG_ZERO>(ts.notZ == 0);
        s.sysFlags.template change<cule::atari::FLAG_CARRY>(ts.C);
        s.sysFlags.set(cule::atari::FLAG_RESERVED);
        s.sysFlags.clear(cule::atari::FLAG_CPU_HALT);
        s.sysFlags.clear(cule::atari::FLAG_CPU_LAST_READ);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_IS_NTSC>(cart.is_ntsc());

        s.ram = reinterpret_cast<uint32_t*>(input_ram + (256 * index));
        uint8_t* ram_ptr = reinterpret_cast<uint8_t*>(s.ram);
        for(size_t i = 0; i < cart.ram_size(); i++)
        {
            ram_ptr[i] = ts.ram[i];
        }

        UPDATE_FIELD(s.riotData, cule::atari::FIELD_RIOT_TIMER, ts.timer);
        UPDATE_FIELD(s.riotData, cule::atari::FIELD_RIOT_SHIFT, ts.intervalShift);
        s.cyclesWhenTimerSet = ts.cyclesWhenTimerSet;
        s.cyclesWhenInterruptReset = ts.cyclesWhenInterruptReset;
        s.tiaFlags.template change<cule::atari::FLAG_RIOT_READ_INT>(ts.timerReadAfterInterrupt != 0);
        UPDATE_FIELD(s.riotData, cule::atari::FIELD_RIOT_DDRA, ts.DDRA);
        ALE_t::set_id(s, cart.game_id());

        s.clockWhenFrameStarted = ts.clockWhenFrameStarted;
        /* ds.clockStartDisplay = TIA_t::clockStartDisplay(s); */
        /* ds.clockStopDisplay = TIA_t::clockStopDisplay(s); */
        s.clockAtLastUpdate = ts.clockAtLastUpdate;
        /* ds.clocksToEndOfScanLine = INT_MAX; */
        /* ds.scanlineCountForLastFrame = INT_MAX; */
        /* ds.currentScanline = TIA_t::currentScanline(s); */
        s.VSYNCFinishClock = ts.VSYNCFinishClock;
        UPDATE_FIELD(s.tiaFlags.asBitField(), cule::atari::FIELD_TIA_ENABLED, ts.enabledObjects);
        /* ds.VSYNC = INT_MAX; */
        s.tiaFlags.template change<cule::atari::FLAG_TIA_VBLANK1>((ts.VBLANK & 0x80) != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_VBLANK2>((ts.VBLANK & 0x02) != 0);
        UPDATE_FIELD(s.PF, cule::atari::FIELD_NUSIZ0_MODE, ts.NUSIZ0 & 0x07);
        UPDATE_FIELD(s.PF, cule::atari::FIELD_NUSIZ0_SIZE, (ts.NUSIZ0 & 0x30) >> 4);
        UPDATE_FIELD(s.PF, cule::atari::FIELD_NUSIZ1_MODE, ts.NUSIZ1 & 0x07);
        UPDATE_FIELD(s.PF, cule::atari::FIELD_NUSIZ1_SIZE, (ts.NUSIZ1 & 0x30) >> 4);
        /* ds.COLUP0 = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUP0); */
        /* ds.COLUP1 = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUP1); */
        /* ds.COLUPF = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUPF); */
        /* ds.COLUBK = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUBK); */
        UPDATE_FIELD(s.PF,  cule::atari::FIELD_CTRLPF, (ts.CTRLPF & 0x30) >> 4);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_CTRLPF>((ts.CTRLPF & 0x01) == 0x01);
        /* ds.playfieldPriorityAndScore = fs.playfieldPriorityAndScore; */
        s.tiaFlags.template change<cule::atari::FLAG_TIA_REFP0>(ts.REFP0 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_REFP1>(ts.REFP0 != 0);
        UPDATE_FIELD(s.PF,  cule::atari::FIELD_PFALL, ts.PF);
        UPDATE_FIELD(s.GRP, cule::atari::FIELD_GRP0, ts.GRP0);
        UPDATE_FIELD(s.GRP, cule::atari::FIELD_GRP1, ts.GRP1);
        UPDATE_FIELD(s.GRP, cule::atari::FIELD_DGRP0, ts.DGRP0);
        UPDATE_FIELD(s.GRP, cule::atari::FIELD_DGRP1, ts.DGRP1);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_ENAM0>(ts.ENAM0 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_ENAM1>(ts.ENAM1 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_ENABL>(ts.ENABL != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_DENABL>(ts.DENABL != 0);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_HMP0, ts.HMP0);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_HMP1, ts.HMP1);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_HMM0, ts.HMM0);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_HMM1, ts.HMM1);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_HMBL, ts.HMBL);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_VDELP0>(ts.VDELP0 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_VDELP1>(ts.VDELP0 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_VDELBL>(ts.VDELBL != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_RESMP0>(ts.RESMP0 != 0);
        s.tiaFlags.template change<cule::atari::FLAG_TIA_RESMP1>(ts.RESMP1 != 0);
        s.collision = ts.collision;
        UPDATE_FIELD(s.POS, cule::atari::FIELD_POSP0, ts.POSP0);
        UPDATE_FIELD(s.POS, cule::atari::FIELD_POSP1, ts.POSP1);
        UPDATE_FIELD(s.POS, cule::atari::FIELD_POSM0, ts.POSM0);
        UPDATE_FIELD(s.POS, cule::atari::FIELD_POSM1, ts.POSM1);
        UPDATE_FIELD(s.HM, cule::atari::FIELD_POSBL, ts.POSBL);
        s.CurrentGRP0 = ts.currentGRP0;
        s.CurrentGRP1 = ts.currentGRP1;
        s.tiaFlags.template change<cule::atari::FLAG_TIA_HMOVE_ALLOW>(ts.HMOVEBlankEnabled != 0);
        s.lastHMOVEClock = ts.lastHMOVEClock;
        /* ds.M0CosmicArkMotionEnabled = INT_MAX; */
        /* ds.M0CosmicArkCounter = INT_MAX; */
        s.tiaFlags.template change<cule::atari::FLAG_TIA_DUMP>(ts.dumpEnabled != 0);
        s.dumpDisabledCycle = ts.dumpDisabledCycle;

        s.bank = ts.bank;
        s.score = ts.score;
        s.tiaFlags.template change<cule::atari::FLAG_ALE_TERMINAL>(ts.terminal != 0);
        s.tiaFlags.template change<cule::atari::FLAG_ALE_STARTED>(ts.started != 0);
    }
};

struct decode_states_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const bool use_cuda,
                    const cule::atari::rom& cart,
                    AtariState* dst_states,
                    cule::atari::state* src_states,
                    cule::atari::frame_state* src_frame_states,
                    uint8_t* src_states_ram) const
    {
        using Environment_t = cule::atari::environment<cule::atari::ROM_2K>;
        using TIA_t = typename Environment_t::TIA_t;

        const size_t index = self.index();

        cule::atari::state& s = src_states[index];
        cule::atari::frame_state& fs = src_frame_states[index];
        AtariState& ds = dst_states[index];

        if(use_cuda)
        {
            s.ram = (uint32_t*) (src_states_ram + (256 * index));
        }

        ds.left_paddle = s.resistance;
        ds.right_paddle = 0;
        ds.frame_number = Environment_t::getFrameNumber(s);
        ds.episode_frame_number = Environment_t::getFrameNumber(s);
        ds.string_length = cart.md5().length();
        ds.save_system = false;
        ds.md5 = cart.md5();

        ds.cycles = s.cpuCycles;

        ds.A = s.A;
        ds.X = s.X;
        ds.Y = s.Y;
        ds.SP = valueOf(s.SP);
        ds.PC = valueOf(s.PC);
        ds.IR = SELECT_FIELD(cule::atari::sys_flag_t(s.sysFlags).asBitField(), cule::atari::FIELD_SYS_INT);

        ds.N = s.sysFlags[cule::atari::FLAG_NEGATIVE];
        ds.V = s.sysFlags[cule::atari::FLAG_OVERFLOW];
        ds.B = s.sysFlags[cule::atari::FLAG_BREAK];
        ds.D = s.sysFlags[cule::atari::FLAG_DECIMAL];
        ds.I = s.sysFlags[cule::atari::FLAG_INTERRUPT_OFF];
        ds.notZ = !s.sysFlags[cule::atari::FLAG_ZERO];
        ds.C = s.sysFlags[cule::atari::FLAG_CARRY];

        ds.executionStatus = 0;

        ds.ram.fill(0);
        uint8_t* ram_ptr = reinterpret_cast<uint8_t*>(s.ram);
        for(size_t i = 0; i < cart.ram_size(); i++)
        {
            ds.ram[i] = ram_ptr[i];
        }

        ds.timer = SELECT_FIELD(s.riotData, cule::atari::FIELD_RIOT_TIMER);
        ds.intervalShift = SELECT_FIELD(s.riotData, cule::atari::FIELD_RIOT_SHIFT);
        ds.cyclesWhenTimerSet = s.cyclesWhenTimerSet;
        ds.cyclesWhenInterruptReset = s.cyclesWhenInterruptReset;
        ds.timerReadAfterInterrupt = s.tiaFlags[cule::atari::FLAG_RIOT_READ_INT];
        ds.DDRA = SELECT_FIELD(s.riotData, cule::atari::FIELD_RIOT_DDRA);
        ds.DDRB = 0;

        ds.clockWhenFrameStarted = s.clockWhenFrameStarted;
        ds.clockStartDisplay = TIA_t::clockStartDisplay(s);
        ds.clockStopDisplay = TIA_t::clockStopDisplay(s);
        ds.clockAtLastUpdate = s.clockAtLastUpdate;
        ds.clocksToEndOfScanLine = INT_MAX;
        ds.scanlineCountForLastFrame = INT_MAX;
        ds.currentScanline = TIA_t::currentScanline(s);
        ds.VSYNCFinishClock = s.VSYNCFinishClock;
        ds.enabledObjects = SELECT_FIELD(cule::atari::tia_flag_t(s.tiaFlags).asBitField(), cule::atari::FIELD_TIA_ENABLED);
        ds.VSYNC = INT_MAX;
        ds.VBLANK = (s.tiaFlags[cule::atari::FLAG_TIA_VBLANK1] << 7) | (s.tiaFlags[cule::atari::FLAG_TIA_VBLANK2] << 1);
        ds.NUSIZ0 = (SELECT_FIELD(s.PF, cule::atari::FIELD_NUSIZ0_SIZE) << 4) | SELECT_FIELD(s.PF, cule::atari::FIELD_NUSIZ0_MODE);
        ds.NUSIZ1 = (SELECT_FIELD(s.PF, cule::atari::FIELD_NUSIZ1_SIZE) << 4) | SELECT_FIELD(s.PF, cule::atari::FIELD_NUSIZ1_MODE);
        ds.COLUP0 = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUP0);
        ds.COLUP1 = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUP1);
        ds.COLUPF = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUPF);
        ds.COLUBK = SELECT_FIELD(fs.Color, cule::atari::FIELD_COLUBK);
        ds.CTRLPF = ((SELECT_FIELD(s.PF,  cule::atari::FIELD_CTRLPF) & 3) << 4) + s.tiaFlags[cule::atari::FLAG_TIA_CTRLPF];
        ds.playfieldPriorityAndScore = fs.playfieldPriorityAndScore;
        ds.REFP0 = s.tiaFlags[cule::atari::FLAG_TIA_REFP0];
        ds.REFP1 = s.tiaFlags[cule::atari::FLAG_TIA_REFP1];
        ds.PF = SELECT_FIELD(s.PF,  cule::atari::FIELD_PFALL);
        ds.GRP0 = SELECT_FIELD(s.GRP, cule::atari::FIELD_GRP0);
        ds.GRP1 = SELECT_FIELD(s.GRP, cule::atari::FIELD_GRP1);
        ds.DGRP0 = SELECT_FIELD(s.GRP, cule::atari::FIELD_DGRP0);
        ds.DGRP1 = SELECT_FIELD(s.GRP, cule::atari::FIELD_DGRP1);
        ds.ENAM0 = s.tiaFlags[cule::atari::FLAG_TIA_ENAM0];
        ds.ENAM1 = s.tiaFlags[cule::atari::FLAG_TIA_ENAM1];
        ds.ENABL = s.tiaFlags[cule::atari::FLAG_TIA_ENABL];
        ds.DENABL = s.tiaFlags[cule::atari::FLAG_TIA_DENABL];
        ds.HMP0 = SELECT_FIELD(s.HM, cule::atari::FIELD_HMP0);
        ds.HMP1 = SELECT_FIELD(s.HM, cule::atari::FIELD_HMP1);
        ds.HMM0 = SELECT_FIELD(s.HM, cule::atari::FIELD_HMM0);
        ds.HMM1 = SELECT_FIELD(s.HM, cule::atari::FIELD_HMM1);
        ds.HMBL = SELECT_FIELD(s.HM, cule::atari::FIELD_HMBL);
        ds.VDELP0 = s.tiaFlags[cule::atari::FLAG_TIA_VDELP0];
        ds.VDELP1 = s.tiaFlags[cule::atari::FLAG_TIA_VDELP1];
        ds.VDELBL = s.tiaFlags[cule::atari::FLAG_TIA_VDELBL];
        ds.RESMP0 = s.tiaFlags[cule::atari::FLAG_TIA_RESMP0];
        ds.RESMP1 = s.tiaFlags[cule::atari::FLAG_TIA_RESMP1];
        ds.collision = s.collision;
        ds.POSP0 = SELECT_FIELD(s.POS, cule::atari::FIELD_POSP0);
        ds.POSP1 = SELECT_FIELD(s.POS, cule::atari::FIELD_POSP1);
        ds.POSM0 = SELECT_FIELD(s.POS, cule::atari::FIELD_POSM0);
        ds.POSM1 = SELECT_FIELD(s.POS, cule::atari::FIELD_POSM1);
        ds.POSBL = SELECT_FIELD(s.HM, cule::atari::FIELD_POSBL);
        ds.currentGRP0 = s.CurrentGRP0;
        ds.currentGRP1 = s.CurrentGRP1;
        ds.HMOVEBlankEnabled = s.tiaFlags[cule::atari::FLAG_TIA_HMOVE_ALLOW];
        ds.lastHMOVEClock = s.lastHMOVEClock;
        ds.M0CosmicArkMotionEnabled = INT_MAX;
        ds.M0CosmicArkCounter = INT_MAX;
        ds.dumpEnabled = s.tiaFlags[cule::atari::FLAG_TIA_DUMP];
        ds.dumpDisabledCycle = s.dumpDisabledCycle;

        // TODO: reward will be incorrect because the internal score has
        // already been updated during the last step
        ds.bank = s.bank;
        ds.reward = cule::atari::ale::getRewards(s);
        ds.score = cule::atari::ale::getScore(s);
        ds.terminal = cule::atari::ale::isTerminal(s);
        ds.started = cule::atari::ale::isStarted(s);
        ds.lives = cule::atari::ale::getLives(s);
        ds.last_lives = 0;

        if(cart.game_id() == cule::atari::games::GAME_TENNIS)
        {
            int16_t *ptr = (int16_t*) &ds.score;
            ds.score = int32_t(ptr[1]);
        }
    }
};

