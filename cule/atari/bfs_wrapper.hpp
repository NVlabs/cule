#pragma once

#include <cule/config.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/state.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{

template<int32_t> struct environment;

class bfs_wrapper
{
public:

    using State_t = state;

    bfs_wrapper(const rom& cart,
                const size_t num_envs,
                const size_t noop_reset_steps = 30);

    size_t image_buffer_size(const size_t num_channels, const bool rescale) const;

    size_t size() const;

    void initialize_ptrs(State_t* states_ptr,
                         frame_state* frame_states_ptr,
                         uint8_t* ram_ptr,
                         uint32_t* tia_update_ptr,
                         uint8_t* frame_ptr,
                         uint32_t* rom_indices_ptr,
                         Action* minimal_actions_ptr,
                         uint32_t* rand_states_ptr,
                         State_t* cached_states_ptr,
                         uint8_t* cached_ram_ptr,
                         frame_state* cached_frame_states_ptr,
                         uint32_t* cached_tia_update_ptr,
                         uint32_t* cache_index_ptr);

    template<typename ExecutionPolicy>
    void reset(ExecutionPolicy&& policy,
               uint32_t* seedBuffer = nullptr);

    template<typename ExecutionPolicy>
    void reset_states(ExecutionPolicy&& policy);

    template<typename ExecutionPolicy>
    void get_states(ExecutionPolicy&& policy,
                    const size_t num_states,
                    const int32_t* indices,
                    State_t* output_states,
                    frame_state* output_frame_states,
                    uint8_t* output_states_ram);

    template<typename ExecutionPolicy>
    void set_states(ExecutionPolicy&& policy,
                    const size_t num_states,
                    const int32_t* indices,
                    const State_t* input_states,
                    const frame_state* input_frame_states,
                    const uint8_t* input_states_ram);

    template<typename ExecutionPolicy>
    void _step(ExecutionPolicy&& policy,
               const bool fire_reset,
               const Action* player_a_buffer,
               const Action* player_b_buffer,
               bool* doneBuffer);

    template<typename ExecutionPolicy>
    void _get_data(ExecutionPolicy&& policy,
                   const bool episodic_life,
                   bool* doneBuffer,
                   float* rewardsBuffer,
                   int32_t* livesBuffer);

    template<typename ExecutionPolicy>
    void step(ExecutionPolicy&& policy,
              const bool fire_reset,
              const size_t num_envs,
              bool* doneBuffer);

    template<typename ExecutionPolicy>
    void get_data(ExecutionPolicy&& policy,
                  const bool episodic_life,
                  const size_t num_envs,
                  const float gamma,
                  bool* doneBuffer,
                  float* rewardsBuffer,
                  int32_t* livesBuffer);

    template<typename ExecutionPolicy>
    void preprocess(ExecutionPolicy&& policy,
                    const bool last_frame,
                    const uint32_t* tiaBuffer,
                    uint8_t* frameBuffer);

    template<typename ExecutionPolicy>
    void generate_frames(ExecutionPolicy&& policy,
                         const bool rescale,
                         const bool last_frame,
                         const size_t num_channels,
                         uint8_t* imageBuffer);

    template<typename ExecutionPolicy>
    void generate_random_actions(ExecutionPolicy&& policy,
                                 Action* actionsBuffer,
                                 const size_t N = 0);

    template<typename ExecutionPolicy>
    void save_images(ExecutionPolicy&& policy,
                     const bool rescale,
                     const size_t num_channels,
                     const size_t frame_index,
                     const uint8_t* imageBuffer);

    rom cart;

    size_t num_envs;
    size_t noop_reset_steps;

    State_t* states_ptr;
    uint8_t* ram_ptr;

    frame_state* frame_states_ptr;
    uint32_t* tia_update_ptr;
    uint8_t* frame_ptr;

    uint32_t* rom_indices_ptr;
    uint32_t* rand_states_ptr;
    uint32_t* cache_index_ptr;

    Action* minimal_actions_ptr;

    State_t* cached_states_ptr;
    uint8_t* cached_ram_ptr;
    uint32_t* cached_tia_update_ptr;

    frame_state* cached_frame_states_ptr;
};

template<template<typename> class Allocator>
class bfs_wrapped_environment : public bfs_wrapper
{
private:

    using super_t = bfs_wrapper;
    using State_t = typename super_t::State_t;

public:

    template<typename U>
    using VecType = agency::vector<U,Allocator<U>>;

    bfs_wrapped_environment(const rom& cart,
                            const size_t num_envs,
                            const size_t noop_reset_steps = 30);

protected:

    VecType<State_t> states_buffer;
    VecType<frame_state> frame_states_buffer;
    VecType<uint8_t> ram_buffer;
    VecType<uint32_t> tia_update_buffer;
    VecType<uint8_t> frame_buffer;

    VecType<uint32_t> rom_indices_buffer;
    VecType<uint32_t> rand_states_buffer;
    VecType<uint32_t> cache_index_buffer;

    VecType<Action> minimal_actions_buffer;

    VecType<State_t> cached_states_buffer;
    VecType<uint8_t> cached_ram_buffer;

    VecType<frame_state> cached_frame_states_buffer;
    VecType<uint32_t> cached_tia_update_buffer;
};

} // end namespace atari
} // end namespace cule

