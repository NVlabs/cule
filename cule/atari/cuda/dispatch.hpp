#pragma once

#include <cule/config.hpp>

#include <cule/cuda/parallel_execution_policy.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/internals.hpp>

#include <cule/atari/cuda/tables.hpp>
#include <cule/atari/cuda/kernels.hpp>

#include <agency/agency.hpp>

namespace cule
{
namespace atari
{
namespace dispatch
{

template<typename Environment,
         typename Wrapper>
void
reset(cule::cuda::parallel_execution_policy& policy,
      Wrapper& wrap,
      uint32_t*)
{
    const size_t BLOCK_SIZE = 1UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    using State_t = typename Wrapper::State_t;

    wrapped_environment<agency::allocator> env(wrap.cart, 1, wrap.noop_reset_steps);
    env.reset(agency::seq);

    assert(env.cached_states_ptr != nullptr);
    assert(env.cached_ram_ptr != nullptr);

    assert(wrap.states_ptr != nullptr);
    assert(wrap.rand_states_ptr != nullptr);
    assert(wrap.cached_states_ptr != nullptr);
    assert(wrap.cached_ram_ptr != nullptr);
    assert(wrap.cached_frame_states_ptr != nullptr);

    cule::atari::initialize_tables(wrap.cart);
    CULE_CUDA_PEEK_AND_SYNC;

    CULE_ERRCHK(cudaMemcpyAsync(wrap.cached_states_ptr,
                                env.cached_states_ptr,
                                sizeof(State_t) * wrap.noop_reset_steps,
                                cudaMemcpyHostToDevice,
                                policy.getStream()));

    CULE_ERRCHK(cudaMemcpyAsync(wrap.cached_ram_ptr,
                                env.cached_ram_ptr,
                                sizeof(uint8_t) * wrap.cart.ram_size() * wrap.noop_reset_steps,
                                cudaMemcpyHostToDevice,
                                policy.getStream()));

    CULE_ERRCHK(cudaMemcpyAsync(wrap.cached_tia_update_ptr,
                                env.cached_tia_update_ptr,
                                sizeof(uint32_t) * ENV_UPDATE_SIZE * wrap.noop_reset_steps,
                                cudaMemcpyHostToDevice,
                                policy.getStream()));

    CULE_ERRCHK(cudaMemcpyAsync(wrap.cached_frame_states_ptr,
                                env.cached_frame_states_ptr,
                                sizeof(frame_state) * wrap.noop_reset_steps,
                                cudaMemcpyHostToDevice,
                                policy.getStream()));

    cule::atari::cuda::initialize_frame_states_kernel<State_t, 128>
    <<<1, 128, 0, policy.getStream()>>>(
        wrap.noop_reset_steps,
        wrap.cached_states_ptr,
        wrap.cached_frame_states_ptr,
        &playfield_accessor(0, 0),
        &player_mask_accessor(0, 0, 0, 0),
        &player_mask_accessor(0, 0, 0, 0),
        &missle_accessor(0, 0, 0, 0),
        &missle_accessor(0, 0, 0, 0),
        &ball_accessor(0, 0, 0));
    CULE_CUDA_PEEK_AND_SYNC;

    cule::atari::cuda::initialize_states_kernel<State_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        wrap.noop_reset_steps,
        wrap.states_ptr,
        wrap.ram_ptr,
        wrap.cached_states_ptr,
        wrap.cached_ram_ptr,
        wrap.rand_states_ptr,
        wrap.frame_states_ptr,
        wrap.cached_frame_states_ptr);
    CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Environment,
         typename Wrapper>
void
reset_states(cule::cuda::parallel_execution_policy& policy,
             Wrapper& wrap)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 1UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    cule::atari::cuda::reset_kernel<State_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        wrap.noop_reset_steps,
        wrap.states_ptr,
        wrap.ram_ptr,
        wrap.cached_states_ptr,
        wrap.cached_ram_ptr,
        wrap.frame_states_ptr,
        wrap.cached_frame_states_ptr,
        wrap.cache_index_ptr,
        wrap.rand_states_ptr);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Environment,
         typename Wrapper>
void
step(cule::cuda::parallel_execution_policy& policy,
     Wrapper& wrap,
     const bool fire_reset,
     const Action* actionsBuffer,
     uint8_t* doneBuffer)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 1UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    cule::atari::cuda::step_kernel<State_t, Environment, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        fire_reset,
        wrap.states_ptr,
        wrap.ram_ptr,
        wrap.tia_update_ptr,
        actionsBuffer,
        doneBuffer);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Environment,
         typename Wrapper>
void
get_data(cule::cuda::parallel_execution_policy& policy,
         Wrapper& wrap,
         const bool episodic_life,
         uint8_t* doneBuffer,
         int32_t* rewardsBuffer,
         int32_t* livesBuffer)
{
    using State_t = typename Wrapper::State_t;
    using ALE_t = typename Environment::ALE_t;

    const size_t BLOCK_SIZE = 256UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    cule::atari::cuda::get_data_kernel<State_t, ALE_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        episodic_life,
        wrap.states_ptr,
        wrap.ram_ptr,
        doneBuffer,
        rewardsBuffer,
        livesBuffer);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Environment,
         typename Wrapper>
void
preprocess(cule::cuda::parallel_execution_policy& policy,
           Wrapper& wrap,
           const uint32_t* tiaBuffer,
           uint8_t* frameBuffer)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 1UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    cule::atari::cuda::process_kernel<State_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        tiaBuffer,
        wrap.cached_tia_update_ptr,
        wrap.cache_index_ptr,
        wrap.states_ptr,
        wrap.frame_states_ptr,
        frameBuffer);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Wrapper>
void
generate_frames(cule::cuda::parallel_execution_policy& policy,
                Wrapper& wrap,
                const bool rescale,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    const size_t BLOCK_SIZE = 1024UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.image_buffer_size(num_channels, rescale) / num_channels) / BLOCK_SIZE);

    if(rescale)
    {
        cule::atari::cuda::apply_rescale_kernel<BLOCK_SIZE>
        <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
            wrap.size(),
            wrap.cart.screen_height(),
            imageBuffer,
            wrap.frame_ptr);
    }
    else
    {
        cule::atari::cuda::apply_palette_kernel<BLOCK_SIZE>
        <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
            wrap.size(),
            wrap.cart.screen_height(),
            num_channels,
            imageBuffer,
            wrap.frame_ptr);
    }
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Wrapper>
void
generate_random_actions(cule::cuda::parallel_execution_policy& policy,
                        Wrapper& wrap,
                        Action* actionsBuffer,
                        const size_t N)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 256UL;
    const size_t NUM_BLOCKS = std::ceil(float(wrap.size()) / BLOCK_SIZE);

    const size_t num_entries = N == 0 ? wrap.size() : N;

    cule::atari::cuda::action_kernel<BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        wrap.size(),
        wrap.cart.minimal_actions().size(),
        num_entries,
        wrap.minimal_actions_ptr,
        wrap.rand_states_ptr,
        actionsBuffer);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Wrapper>
void
get_states(cule::cuda::parallel_execution_policy& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           typename Wrapper::State_t* output_states,
           frame_state* output_frame_states,
           uint8_t* output_states_ram)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 128UL;
    const size_t NUM_BLOCKS = std::ceil(float(num_states) / BLOCK_SIZE);

    cule::atari::cuda::get_states_kernel<State_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        num_states,
        indices,
        wrap.states_ptr,
        wrap.frame_states_ptr,
        output_states,
        output_frame_states,
        wrap.ram_ptr,
        output_states_ram);
    // CULE_CUDA_PEEK_AND_SYNC;
}

template<typename Wrapper>
void
set_states(cule::cuda::parallel_execution_policy& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           const typename Wrapper::State_t* input_states,
           const frame_state* input_frame_states,
           const uint8_t* input_states_ram)
{
    using State_t = typename Wrapper::State_t;

    const size_t BLOCK_SIZE = 128UL;
    const size_t NUM_BLOCKS = std::ceil(float(num_states) / BLOCK_SIZE);

    cule::atari::cuda::set_states_kernel<State_t, BLOCK_SIZE>
    <<<NUM_BLOCKS, BLOCK_SIZE, 0, policy.getStream()>>>(
        num_states,
        indices,
        wrap.states_ptr,
        wrap.frame_states_ptr,
        input_states,
        input_frame_states,
        wrap.ram_ptr,
        input_states_ram);
    // CULE_CUDA_PEEK_AND_SYNC;
}

} // end namespace dispatch
} // end namespace atari
} // end namespace cule

