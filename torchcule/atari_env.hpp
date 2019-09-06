#pragma once

#include <cule/atari/flags.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/wrapper.hpp>

#include <cule/atari/games/detail/attributes.hpp>

#include <torchcule/atari_state.hpp>

class AtariEnv : public cule::atari::wrapper
{
    private:
      using super_t = cule::atari::wrapper;

    public:
      AtariEnv(const cule::atari::rom& cart,
               const size_t num_envs,
               const size_t noop_reset_steps);

      ~AtariEnv();

      void reset(uint32_t* seedBuffer);

      void reset_states();

      void get_states(const size_t num_states,
                      const int32_t* indices,
                      AtariState* states);

      void set_states(const size_t num_states,
                      const int32_t* indices,
                      const AtariState* states);

      void step(const bool fire_reset,
                const cule::atari::Action* actionsBuffer,
                uint8_t* doneBuffer);

      void two_step(const cule::atari::Action* playerABuffer,
                    const cule::atari::Action* playerBBuffer);

      void get_data(const bool episodic_life,
                    uint8_t* doneBuffer,
                    int32_t* rewardsBuffer,
                    int32_t* livesBuffer);

      void generate_frames(const bool rescale,
                           const bool last_frame,
                           const size_t num_channels,
                           uint8_t* imageBuffer);

      void generate_random_actions(cule::atari::Action* actionBuffer);

      void set_cuda(const bool use_cuda, const int32_t gpu_id);

      size_t state_size();

      size_t frame_state_size();

      size_t tia_update_size();

      void sync_other_stream(cudaStream_t& stream);

      void sync_this_stream(cudaStream_t& stream);

      template<typename ExecutionPolicy>
      ExecutionPolicy& get_policy();

    private:
      void* cule_par;
      size_t num_channels;
      bool rescale;
      bool use_cuda;
      int32_t gpu_id;
};

