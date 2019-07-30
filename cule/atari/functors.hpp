#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/m6502.hpp>
#include <cule/atari/paddles.hpp>
#include <cule/atari/palettes.hpp>
#include <cule/atari/png.hpp>
#include <cule/atari/prng.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{

template<typename Environment_t>
struct initialize_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const uint32_t* rom_indices,
                    const rom* cart,
                    uint8_t* ram_buffer,
                    State_t* states_buffer,
                    frame_state* frame_states_buffer,
                    uint32_t* rand_states_buffer,
                    const uint32_t* seed_buffer) const
    {
        using Accessor_t = typename Environment_t::Accessor_t;
        using ALE_t = typename Environment_t::ALE_t;
        using Controller_t = typename Environment_t::Controller_t;

        assert(rom_indices != nullptr);
        assert(cart != nullptr);
        assert(ram_buffer != nullptr);
        assert(states_buffer != nullptr);
        assert(frame_states_buffer != nullptr);
        assert(rand_states_buffer != nullptr);
        assert(seed_buffer != nullptr);

        const bool use_paddles = cart->use_paddles();
        const bool swap_paddles = cart->swap_paddles() || cart->swap_ports();
        const bool left_difficulty_B = cart->player_left_difficulty_B();
        const bool right_difficulty_B = cart->player_right_difficulty_B();
        const bool is_ntsc = cart->is_ntsc();
        const bool hmove_blanks = cart->allow_hmove_blanks();
        const games::GAME_TYPE game_id = cart->game_id();
        const uint8_t * rom_buffer = cart->data();

        prng gen(rand_states_buffer[self.index()]);
        gen.initialize(seed_buffer[self.index()]);

        State_t& s = states_buffer[self.index()];

        s.rand = gen.sample();

        s.ram = (uint32_t*)(ram_buffer + (cart->ram_size() * self.index()));
        s.rom = rom_buffer;
        s.tia_update_buffer = nullptr;

        Accessor_t::initialize(s);

        // update controller settings
        Controller_t::set_flags(s, use_paddles, swap_paddles, left_difficulty_B, right_difficulty_B);

        // set the game id
        ALE_t::set_id(s, game_id);

        s.tiaFlags.template change<FLAG_TIA_IS_NTSC>(is_ntsc);
        s.tiaFlags.template change<FLAG_TIA_HMOVE_ALLOW>(hmove_blanks);

        s.tiaFlags.set(FLAG_ALE_TERMINAL);
        s.tiaFlags.set(FLAG_ALE_STARTED);
        s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);

        const int32_t num_actions = 1;
        const Action starting_action = ACTION_FIRE;

        Environment_t::setStartNumber(s, num_actions);
        Environment_t::setStartAction(s, starting_action);

        frame_state& fs = frame_states_buffer[self.index()];
        fs.tiaFlags.template change<FLAG_TIA_IS_NTSC>(is_ntsc);
        fs.tiaFlags.template change<FLAG_TIA_HMOVE_ALLOW>(hmove_blanks);
    }
};

template<typename Environment_t>
struct reset_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const size_t ram_size,
                    const size_t noop_reset_steps,
                    State_t* states_buffer,
                    const State_t* cached_states_buffer,
                    const uint8_t* cached_ram_buffer,
                    frame_state* frame_states_buffer,
                    frame_state* cached_frame_states_buffer,
                    uint32_t* cache_index_buffer,
                    uint32_t* rand_states_buffer) const
    {
        State_t& s = states_buffer[self.index()];

        if(s.tiaFlags[FLAG_ALE_TERMINAL])
        {
            const size_t NUM_RAM_INTS = ram_size / sizeof(uint32_t);
            prng gen(rand_states_buffer[self.index()]);
            const size_t cache_index = gen.sample() % noop_reset_steps;

            uint32_t* ram = s.ram;
            s = cached_states_buffer[cache_index];
            s.tiaFlags.set(FLAG_ALE_TERMINAL);
            s.ram = ram;
            cache_index_buffer[self.index()] = cache_index;

            uint32_t* ram_int = ((uint32_t*) cached_ram_buffer) + (NUM_RAM_INTS * cache_index);

            for(size_t i = 0; i < NUM_RAM_INTS; i++)
            {
                ram[i] = ram_int[i];
            }

            frame_states_buffer[self.index()] = cached_frame_states_buffer[cache_index];
        }
    }
};

template<typename Environment_t>
struct step_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const bool fire_reset,
                    State_t* states_buffer,
                    uint32_t* tia_update_buffer,
                    Action* action_buffer,
                    uint8_t* done_buffer) const
    {
        if((done_buffer != nullptr) && done_buffer[self.index()])
        {
            return;
        }

        State_t& s = states_buffer[self.index()];

        s.tia_update_buffer = nullptr;
        if(tia_update_buffer != nullptr)
        {
            s.tia_update_buffer = tia_update_buffer + (ENV_UPDATE_SIZE * self.index());
        }

        Action player_a_action = ACTION_NOOP;
        Action player_b_action = ACTION_NOOP;

        if(action_buffer != nullptr)
        {
            player_a_action = action_buffer[self.index()];
        }

        if(fire_reset && s.tiaFlags[FLAG_ALE_LOST_LIFE])
        {
            player_a_action = ACTION_FIRE;
            s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);
        }

        Environment_t::act(s, player_a_action, player_b_action);
    }
};

template<typename Environment_t>
struct get_data_functor
{
    template<class Agent, typename State_t>
    void operator()(Agent& self,
                    const bool episodic_life,
                    State_t* states_buffer,
                    uint8_t* done_buffer,
                    int32_t* rewards_buffer,
                    int32_t* lives_buffer)
    {
        using ALE_t = typename Environment_t::ALE_t;

        if((done_buffer != nullptr) && done_buffer[self.index()])
        {
            return;
        }

        State_t& s = states_buffer[self.index()];

        const uint32_t old_lives = lives_buffer[self.index()];
        const uint32_t new_lives = ALE_t::getLives(s);
        lives_buffer[self.index()] = new_lives;

        const bool lost_life = new_lives < old_lives;
        s.tiaFlags.template change<FLAG_ALE_LOST_LIFE>(lost_life);

        rewards_buffer[self.index()] += ALE_t::getRewards(s);
        done_buffer[self.index()] |= s.tiaFlags[FLAG_ALE_TERMINAL] || (episodic_life && lost_life);

        s.score = ALE_t::getScore(s);
    }
};

template<typename Environment_t>
struct preprocess_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const uint32_t* tia_update_buffer,
                    const uint32_t* cached_tia_update_buffer,
                    State_t* states_buffer,
                    const uint32_t* cache_index_buffer,
                    frame_state* frame_states_buffer,
                    uint8_t* frame_buffer)
    {
        CULE_ASSERT(tia_update_buffer != nullptr);
        CULE_ASSERT(states_buffer != nullptr);
        CULE_ASSERT(frame_states_buffer != nullptr);

        State_t& s = states_buffer[self.index()];
        frame_state& fs = frame_states_buffer[self.index()];
        fs.srcBuffer = tia_update_buffer + (ENV_UPDATE_SIZE * self.index());

        const bool is_terminal = s.tiaFlags[FLAG_ALE_TERMINAL];
        const bool is_started  = s.tiaFlags[FLAG_ALE_STARTED];
        if(is_started && is_terminal)
        {
            CULE_ASSERT(cached_tia_update_buffer != nullptr);
            CULE_ASSERT(cache_index_buffer != nullptr);

            s.tiaFlags.clear(FLAG_ALE_TERMINAL);
            fs.srcBuffer = cached_tia_update_buffer + (cache_index_buffer[self.index()] * ENV_UPDATE_SIZE);
        }
        fs.framePointer = frame_buffer == nullptr ? nullptr : &frame_buffer[self.index() * 300 * SCREEN_WIDTH];

        preprocess::state_to_buffer(fs);
    }
};

struct generate_frame_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const size_t num_channels,
                    const size_t screen_height,
                    const bool rescale,
                    const uint8_t* frame_buffer,
                    uint8_t* image_buffer)
    {
        const uint8_t* framePointer = &frame_buffer[self.index() * 300 * SCREEN_WIDTH];

        if(rescale)
        {
            uint8_t * image_buffer_temp = &image_buffer[self.index() * num_channels * SCALED_SCREEN_SIZE];
            apply_rescale(screen_height, num_channels, image_buffer_temp, framePointer);
        }
        else
        {
            uint8_t * image_buffer_temp = &image_buffer[self.index() * num_channels * screen_height * SCREEN_WIDTH];
            apply_palette(screen_height, num_channels, image_buffer_temp, framePointer);
        }
    }
};

struct get_states_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const int32_t* indices,
                    const State_t* states_buffer,
                    const frame_state* frame_states_buffer,
                    State_t* input_states_buffer,
                    frame_state* input_frame_states_buffer) const
    {
        const size_t index = indices[self.index()];
        input_states_buffer[self.index()] = states_buffer[index];
        input_frame_states_buffer[self.index()] = frame_states_buffer[index];
    }
};

struct set_states_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const int32_t* indices,
                    State_t* states_buffer,
                    frame_state*,
                    const State_t* input_states_buffer,
                    const frame_state*) const
    {
        const size_t index = indices[self.index()];

        const State_t& s = input_states_buffer[self.index()];
        State_t& t = states_buffer[index];

        // frame_states_buffer[index] = input_frame_states_buffer[self.index()];

        t.A = s.A;
        t.X = s.X;
        t.Y = s.Y;
        t.SP = s.SP;
        t.PC = s.PC;
        // t.addr = s.addr;
        // t.value = s.value;
        // t.noise = s.noise;

        t.cpuCycles = s.cpuCycles;
        t.bank = s.bank;

        t.resistance = s.resistance;

        t.GRP = s.GRP;
        t.HM = s.HM;
        t.PF = s.PF;
        t.POS = s.POS;
        t.CurrentGRP0 = s.CurrentGRP0;
        t.CurrentGRP1 = s.CurrentGRP1;

        t.collision = s.collision;
        t.clockWhenFrameStarted = s.clockWhenFrameStarted;
        t.clockAtLastUpdate = s.clockAtLastUpdate;
        t.dumpDisabledCycle = s.dumpDisabledCycle;
        t.VSYNCFinishClock = s.VSYNCFinishClock;
        t.lastHMOVEClock = s.lastHMOVEClock;

        t.riotData = s.riotData;
        t.cyclesWhenTimerSet = s.cyclesWhenTimerSet;
        t.cyclesWhenInterruptReset = s.cyclesWhenInterruptReset;

        t.sysFlags.template change<FLAG_NEGATIVE>(s.sysFlags[FLAG_NEGATIVE]);
        t.sysFlags.template change<FLAG_OVERFLOW>(s.sysFlags[FLAG_OVERFLOW]);
        t.sysFlags.template change<FLAG_BREAK>(s.sysFlags[FLAG_BREAK]);
        t.sysFlags.template change<FLAG_DECIMAL>(s.sysFlags[FLAG_DECIMAL]);
        t.sysFlags.template change<FLAG_INTERRUPT_OFF>(s.sysFlags[FLAG_INTERRUPT_OFF]);
        t.sysFlags.template change<FLAG_ZERO>(s.sysFlags[FLAG_ZERO]);
        t.sysFlags.template change<FLAG_CARRY>(s.sysFlags[FLAG_CARRY]);

        t.tiaFlags = s.tiaFlags;

        t.frameData = s.frameData;
        t.score = s.score;

        for(size_t i = 0; i < (128 / sizeof(uint32_t)); i++)
        {
            t.ram[i] = s.ram[i];
        }
    }
};

struct random_actions_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const size_t minimal_actions_size,
                    const size_t num_entries,
                    const Action* minimal_actions_ptr,
                    uint32_t* rand_states_ptr,
                    Action* actionsBuffer) const
    {
        assert(minimal_actions_ptr != nullptr);
        assert(rand_states_ptr != nullptr);
        assert(actionsBuffer != nullptr);

        prng gen(rand_states_ptr[self.index()]);

        for(size_t i = self.index(); i < num_entries; i += self.group_size())
        {
            actionsBuffer[i] = minimal_actions_ptr[gen.sample() % minimal_actions_size];
        }
    }
};

struct png_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const size_t frame_index,
                    const size_t num_channels,
                    const size_t screen_height,
                    const bool rescale,
                    const uint8_t* image_buffer) const
    {
        const uint8_t * buffer = &image_buffer[self.index() * num_channels * screen_height];
        const std::string filename = get_frame_name(self.index(), frame_index);

        generate_png(buffer, filename, num_channels, rescale, !rescale);
    }
};

} // end namespace atari
} // end namespace cule

