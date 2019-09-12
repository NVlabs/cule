#include <cule/cule.hpp>
#include <cule/cuda.hpp>

#include <torchcule/atari_env.hpp>
#include <torchcule/atari_state.hpp>
#include <torchcule/atari_state.cpp>

using cule_policy = cule::cuda::parallel_execution_policy;

AtariEnv::
AtariEnv(const cule::atari::rom& cart,
         const size_t num_envs,
         const size_t noop_reset_steps)
    : super_t(cart, num_envs, noop_reset_steps),
      use_cuda(false),
      gpu_id(-1),
      cule_par(new agency::parallel_execution_policy())
{
}

AtariEnv::
~AtariEnv()
{
    if(use_cuda)
    {
        delete &get_policy<cule_policy>();
    }
    else
    {
        delete &get_policy<agency::parallel_execution_policy>();
    }
}

void
AtariEnv::
reset(uint32_t* seedBuffer)
{
    if(use_cuda)
    {
        super_t::reset(get_policy<cule_policy>(), seedBuffer);
    }
    else
    {
        super_t::reset(get_policy<agency::parallel_execution_policy>(), seedBuffer);
    }
}

void
AtariEnv::
reset_states()
{
    if(use_cuda)
    {
        super_t::reset_states(get_policy<cule_policy>());
    }
    else
    {
        super_t::reset_states(get_policy<agency::parallel_execution_policy>());
    }
}

void
AtariEnv::
get_states(const size_t num_states,
           const int32_t* indices_ptr,
           AtariState* states_ptr)
{
    agency::vector<cule::atari::state> output_states(num_states);
    agency::vector<cule::atari::frame_state> output_frame_states(num_states);
    agency::vector<uint8_t> output_states_ram(256 * num_states);

    if(use_cuda)
    {
        agency::vector<cule::atari::state, agency::cuda::allocator<cule::atari::state>> output_states_gpu(num_states);
        agency::vector<cule::atari::frame_state, agency::cuda::allocator<cule::atari::frame_state>> output_frame_states_gpu(num_states);
        agency::vector<uint8_t, agency::cuda::allocator<uint8_t>> output_states_ram_gpu(256 * num_states);
        agency::vector<int32_t, agency::cuda::allocator<int32_t>> indices_gpu(indices_ptr, indices_ptr + num_states);

        super_t::get_states(get_policy<cule_policy>(), num_states, indices_gpu.data(), output_states_gpu.data(), output_frame_states_gpu.data(), output_states_ram_gpu.data());
        get_policy<cule_policy>().sync();

        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_states_gpu.begin(), output_states_gpu.end(), output_states.begin());
        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_frame_states_gpu.begin(), output_frame_states_gpu.end(), output_frame_states.begin());
        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_states_ram_gpu.begin(), output_states_ram_gpu.end(), output_states_ram.begin());
        get_policy<cule_policy>().sync();
    }
    else
    {
        super_t::get_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr, output_states.data(), output_frame_states.data(), nullptr);
    }

    agency::bulk_invoke(get_policy<agency::parallel_execution_policy>()(num_states),
                        decode_states_functor{},
                        use_cuda,
                        this->cart,
                        states_ptr,
                        output_states.data(),
                        output_frame_states.data(),
                        output_states_ram.data());
}

void
AtariEnv::
set_states(const size_t num_states,
           const int32_t* indices_ptr,
           const AtariState* states_ptr)
{
    agency::vector<cule::atari::state> input_states(num_states);
    agency::vector<cule::atari::frame_state> input_frame_states(num_states);
    agency::vector<uint8_t> input_states_ram(256 * num_states);

    agency::bulk_invoke(agency::seq(num_states),
                        encode_states_functor{},
                        this->cart,
                        states_ptr,
                        input_states.data(),
                        input_frame_states.data(),
                        input_states_ram.data());

    if(use_cuda)
    {
        agency::vector<cule::atari::state, agency::cuda::allocator<cule::atari::state>> input_states_gpu(input_states.begin(), input_states.end());
        agency::vector<cule::atari::frame_state, agency::cuda::allocator<cule::atari::frame_state>> input_frame_states_gpu(input_frame_states.begin(), input_frame_states.end());
        agency::vector<uint8_t, agency::cuda::allocator<uint8_t>> input_states_ram_gpu(input_states_ram.begin(), input_states_ram.end());
        agency::vector<int32_t, agency::cuda::allocator<int32_t>> indices_gpu(indices_ptr, indices_ptr + num_states);

        super_t::set_states(get_policy<cule_policy>(), num_states, indices_gpu.data(), input_states_gpu.data(),
                            input_frame_states_gpu.data(), input_states_ram_gpu.data());
        get_policy<cule_policy>().sync();
    }
    else
    {
        super_t::set_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr,
                            input_states.data(), input_frame_states.data(), input_states_ram.data());
    }
}

void
AtariEnv::
step(const bool fire_reset,
     const cule::atari::Action* actionsBuffer,
     uint8_t* doneBuffer)
{
    if(use_cuda)
    {
        super_t::step(get_policy<cule_policy>(), fire_reset, actionsBuffer, doneBuffer);
    }
    else
    {
        super_t::step(get_policy<agency::parallel_execution_policy>(), fire_reset, actionsBuffer, doneBuffer);
    }
}

void
AtariEnv::
get_data(const bool episodic_life,
         uint8_t* doneBuffer,
         int32_t* rewardsBuffer,
         int32_t* livesBuffer)
{
    if(use_cuda)
    {
        super_t::get_data(get_policy<cule_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
    else
    {
        super_t::get_data(get_policy<agency::parallel_execution_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
}

void
AtariEnv::
two_step(const cule::atari::Action* playerABuffer,
         const cule::atari::Action* playerBBuffer)
{
    if(use_cuda)
    {
        super_t::two_step(get_policy<cule_policy>(), playerABuffer, playerBBuffer);
    }
    else
    {
        super_t::two_step(get_policy<agency::parallel_execution_policy>(), playerABuffer, playerBBuffer);
    }
}

void
AtariEnv::
generate_frames(const bool rescale,
                const bool last_frame,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    if(use_cuda)
    {
        super_t::generate_frames(get_policy<cule_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
    else
    {
        super_t::generate_frames(get_policy<agency::parallel_execution_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
}

void
AtariEnv::
generate_random_actions(cule::atari::Action* actionBuffer)
{
    if(use_cuda)
    {
        super_t::generate_random_actions(get_policy<cule_policy>(), actionBuffer);
    }
    else
    {
        super_t::generate_random_actions(get_policy<agency::parallel_execution_policy>(), actionBuffer);
    }
}

void
AtariEnv::
sync_other_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_other_stream(stream);
    }
}

void
AtariEnv::
sync_this_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_this_stream(stream);
    }
}

void
AtariEnv::
set_cuda(const bool use_cuda, const int32_t gpu_id)
{
    if(this->use_cuda != use_cuda)
    {
        this->use_cuda = use_cuda;

        if(use_cuda)
        {
            this->gpu_id = gpu_id;
            CULE_ERRCHK(cudaSetDevice(gpu_id));
            delete &get_policy<agency::parallel_execution_policy>();
            cule_par = new cule_policy();
        }
        else
        {
            this->gpu_id = -1;
            delete &get_policy<cule_policy>();
            cule_par = new agency::parallel_execution_policy();
        }
    }
}

template<typename ExecutionPolicy>
ExecutionPolicy&
AtariEnv::
get_policy()
{
    assert(cule_par != nullptr);
    if(gpu_id != -1)
    {
        CULE_ERRCHK(cudaSetDevice(gpu_id));
    }

    return *reinterpret_cast<ExecutionPolicy*>(cule_par);
}

size_t
AtariEnv::
state_size()
{
    return sizeof(cule::atari::state);
}

size_t
AtariEnv::
frame_state_size()
{
    return sizeof(cule::atari::frame_state);
}

size_t
AtariEnv::
tia_update_size()
{
    return cule::atari::ENV_UPDATE_SIZE;
}

#include <cule/atari/rom.cpp>
#include <cule/atari/wrapper.cpp>

