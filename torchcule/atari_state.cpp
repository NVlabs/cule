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
        const AtariState& s = src_states[self.index()];
        cule::atari::state& t = dst_states[self.index()];

        t.bank = s.bank;
        t.clockAtLastUpdate = s.clockAtLastUpdate;
        t.clockWhenFrameStarted = s.clockWhenFrameStarted;
        t.collision = s.collision;
        t.cpuCycles = s.cycles;
        t.cyclesWhenTimerSet = s.cyclesWhenTimerSet;
        t.cyclesWhenInterruptReset = s.cyclesWhenInterruptReset;
        t.dumpDisabledCycle = s.dumpDisabledCycle;
        t.lastHMOVEClock = s.lastHMOVEClock;
        t.resistance = s.left_paddle;
        t.score = s.score;

        t.ram = reinterpret_cast<uint32_t*>(input_ram + (256 * self.index()));
        uint8_t* ram_ptr = reinterpret_cast<uint8_t*>(t.ram);
        for(size_t i = 0; i < cart.ram_size(); i++)
        {
            ram_ptr[i] = s.ram[i];
        }

        t.A = s.A;
        t.X = s.X;
        t.Y = s.Y;
        t.SP = s.SP;
        t.PC = s.PC;
        t.CurrentGRP0 = s.currentGRP0;
        t.CurrentGRP1 = s.currentGRP1;
        t.M0CosmicArkCounter = s.M0CosmicArkCounter;
        t.VSYNCFinishClock = s.VSYNCFinishClock;

        t.sysFlags.template change<cule::atari::FLAG_CARRY>(s.C);
        t.sysFlags.template change<cule::atari::FLAG_ZERO>(s.notZ == 0);
        t.sysFlags.template change<cule::atari::FLAG_INTERRUPT_OFF>(s.I);
        t.sysFlags.template change<cule::atari::FLAG_DECIMAL>(s.D);
        t.sysFlags.template change<cule::atari::FLAG_BREAK>(s.B);
        t.sysFlags.set(cule::atari::FLAG_RESERVED);
        t.sysFlags.template change<cule::atari::FLAG_OVERFLOW>(s.V);
        t.sysFlags.template change<cule::atari::FLAG_NEGATIVE>(s.N);
        t.sysFlags.template change<cule::atari::FLAG_CON_PADDLES>(cart.use_paddles());
        t.sysFlags.template change<cule::atari::FLAG_CON_SWAP>(cart.swap_paddles() || cart.swap_ports());
        t.sysFlags.set(cule::atari::FLAG_SW_RESET_OFF);
        t.sysFlags.set(cule::atari::FLAG_SW_SELECT_OFF);
        t.sysFlags.set(cule::atari::FLAG_SW_UNUSED1);
        t.sysFlags.set(cule::atari::FLAG_SW_COLOR);
        t.sysFlags.set(cule::atari::FLAG_SW_UNUSED2);
        t.sysFlags.set(cule::atari::FLAG_SW_UNUSED3);
        t.sysFlags.template change<cule::atari::FLAG_SW_LEFT_DIFFLAG_A>(!cart.player_left_difficulty_B());
        t.sysFlags.template change<cule::atari::FLAG_SW_RIGHT_DIFFLAG_A>(!cart.player_right_difficulty_B());
        t.sysFlags.set(cule::atari::FLAG_CPU_WRITE_BACK);
        t.sysFlags.template change<cule::atari::FLAG_INT_NMI>((s.executionStatus & 0x08) != 0);
        t.sysFlags.template change<cule::atari::FLAG_INT_IRQ>((s.executionStatus & 0x04) != 0);
        t.sysFlags.template change<cule::atari::FLAG_CPU_ERROR>((s.executionStatus & 0x02) != 0);
        // t.sysFlags.template change<cule::atari::FLAG_CPU_HALT>((s.executionStatus & 0x01) != 0);

        t.tiaFlags.template change<cule::atari::FLAG_TIA_IS_NTSC>(cart.is_ntsc());
        t.tiaFlags.template change<cule::atari::FLAG_TIA_VBLANK1>((s.VBLANK & 0x80) != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_VBLANK2>((s.VBLANK & 0x02) != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_CTRLPF>((s.CTRLPF & 0x01) != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_REFP0>(s.REFP0 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_REFP1>(s.REFP1 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_ENAM0>(s.ENAM0 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_ENAM1>(s.ENAM1 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_ENABL>(s.ENABL != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_DENABL>(s.DENABL != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_VDELP0>(s.VDELP0 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_VDELP1>(s.VDELP1 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_VDELBL>(s.VDELBL != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_RESMP0>(s.RESMP0 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_RESMP1>(s.RESMP1 != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_HMOVE_ALLOW>(cart.allow_hmove_blanks());
        t.tiaFlags.template change<cule::atari::FLAG_TIA_HMOVE_ENABLE>(s.HMOVEBlankEnabled);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_COSMIC_ARK>(s.M0CosmicArkMotionEnabled);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_DUMP>(s.dumpEnabled != 0);
        t.tiaFlags.template change<cule::atari::FLAG_TIA_Y_SHIFT>(cart.game_id() != cule::atari::games::GAME_UP_N_DOWN);
        t.tiaFlags.template change<cule::atari::FLAG_RIOT_READ_INT>(s.timerReadAfterInterrupt != 0);
        t.tiaFlags.template change<cule::atari::FLAG_ALE_TERMINAL>(s.terminal != 0);
        t.tiaFlags.template change<cule::atari::FLAG_ALE_STARTED>(s.started != 0);
        UPDATE_FIELD(t.tiaFlags.asBitField(), cule::atari::FIELD_TIA_ENABLED, s.enabledObjects);

        UPDATE_FIELD(t.frameData, cule::atari::FIELD_FRAME_NUMBER, cule::atari::ENV_BASE_FRAMES + 10 + s.frame_number);

        UPDATE_FIELD(t.riotData, cule::atari::FIELD_RIOT_GAME, cart.game_id());
        UPDATE_FIELD(t.riotData, cule::atari::FIELD_RIOT_TIMER, s.timer);
        UPDATE_FIELD(t.riotData, cule::atari::FIELD_RIOT_SHIFT, s.intervalShift);
        UPDATE_FIELD(t.riotData, cule::atari::FIELD_RIOT_DDRA, s.DDRA);

        UPDATE_FIELD(t.PF, cule::atari::FIELD_NUSIZ0_MODE, s.NUSIZ0 & 0x07);
        UPDATE_FIELD(t.PF, cule::atari::FIELD_NUSIZ0_SIZE, (s.NUSIZ0 & 0x30) >> 4);
        UPDATE_FIELD(t.PF, cule::atari::FIELD_NUSIZ1_MODE, s.NUSIZ1 & 0x07);
        UPDATE_FIELD(t.PF, cule::atari::FIELD_NUSIZ1_SIZE, (s.NUSIZ1 & 0x30) >> 4);
        UPDATE_FIELD(t.PF, cule::atari::FIELD_CTRLPF, (s.CTRLPF & 0x30) >> 4);
        UPDATE_FIELD(t.PF, cule::atari::FIELD_PFALL, s.PF);

        UPDATE_FIELD(t.GRP, cule::atari::FIELD_GRP0, s.GRP0);
        UPDATE_FIELD(t.GRP, cule::atari::FIELD_GRP1, s.GRP1);
        UPDATE_FIELD(t.GRP, cule::atari::FIELD_DGRP0, s.DGRP0);
        UPDATE_FIELD(t.GRP, cule::atari::FIELD_DGRP1, s.DGRP1);

        UPDATE_FIELD(t.HM, cule::atari::FIELD_HMP0, s.HMP0);
        UPDATE_FIELD(t.HM, cule::atari::FIELD_HMP1, s.HMP1);
        UPDATE_FIELD(t.HM, cule::atari::FIELD_HMM0, s.HMM0);
        UPDATE_FIELD(t.HM, cule::atari::FIELD_HMM1, s.HMM1);
        UPDATE_FIELD(t.HM, cule::atari::FIELD_HMBL, s.HMBL);

        UPDATE_FIELD(t.POS, cule::atari::FIELD_POSP0, s.POSP0);
        UPDATE_FIELD(t.POS, cule::atari::FIELD_POSP1, s.POSP1);
        UPDATE_FIELD(t.POS, cule::atari::FIELD_POSM0, s.POSM0);
        UPDATE_FIELD(t.POS, cule::atari::FIELD_POSM1, s.POSM1);
        UPDATE_FIELD(t.HM , cule::atari::FIELD_POSBL, s.POSBL);
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
        ds.CTRLPF = (SELECT_FIELD(s.PF,  cule::atari::FIELD_CTRLPF) << 4) + s.tiaFlags[cule::atari::FLAG_TIA_CTRLPF];
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
        ds.HMOVEBlankEnabled = s.tiaFlags[cule::atari::FLAG_TIA_HMOVE_ENABLE];
        ds.lastHMOVEClock = s.lastHMOVEClock;
        ds.M0CosmicArkMotionEnabled = s.tiaFlags[cule::atari::FLAG_TIA_COSMIC_ARK];
        ds.M0CosmicArkCounter = s.M0CosmicArkCounter;
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

