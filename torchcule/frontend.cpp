#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cuda_runtime.h>

#include <string>

#include <torchcule/atari_env.hpp>

namespace py = pybind11;

using AtariAction = cule::atari::Action;
using AtariRom = cule::atari::rom;
using AtariRomFormat = cule::atari::ROM_FORMAT;

PYBIND11_MODULE(torchcule_atari, m) {
    py::enum_<AtariAction>(m, "AtariAction", py::arithmetic())
    .value("NOOP", AtariAction::ACTION_NOOP)
    .value("RIGHT", AtariAction::ACTION_RIGHT)
    .value("LEFT", AtariAction::ACTION_LEFT)
    .value("DOWN", AtariAction::ACTION_DOWN)
    .value("UP", AtariAction::ACTION_UP)
    .value("FIRE", AtariAction::ACTION_FIRE)
    .value("RESET", AtariAction::ACTION_RESET)
    .value("UPRIGHT", AtariAction::ACTION_UPRIGHT)
    .value("UPLEFT", AtariAction::ACTION_UPLEFT)
    .value("DOWNRIGHT", AtariAction::ACTION_DOWNRIGHT)
    .value("DOWNLEFT", AtariAction::ACTION_DOWNLEFT)
    .value("UPFIRE", AtariAction::ACTION_UPFIRE)
    .value("RIGHTFIRE", AtariAction::ACTION_RIGHTFIRE)
    .value("LEFTFIRE", AtariAction::ACTION_LEFTFIRE)
    .value("DOWNFIRE", AtariAction::ACTION_DOWNFIRE)
    .value("UPRIGHTFIRE", AtariAction::ACTION_UPRIGHTFIRE)
    .value("UPLEFTFIRE", AtariAction::ACTION_UPLEFTFIRE)
    .value("DOWNRIGHTFIRE", AtariAction::ACTION_DOWNRIGHTFIRE)
    .value("DOWNLEFTFIRE", AtariAction::ACTION_DOWNLEFTFIRE)
    .export_values()
    ;

    py::enum_<AtariRomFormat>(m, "AtariRomFormat", py::arithmetic())
    .value("ROM_2K", AtariRomFormat::ROM_2K)
    .value("ROM_4K", AtariRomFormat::ROM_4K)
    .value("ROM_CV", AtariRomFormat::ROM_CV)
    .value("ROM_F8SC", AtariRomFormat::ROM_F8SC)
    .value("ROM_E0", AtariRomFormat::ROM_E0)
    .value("ROM_3E", AtariRomFormat::ROM_3E)
    .value("ROM_3F", AtariRomFormat::ROM_3F)
    .value("ROM_UA", AtariRomFormat::ROM_UA)
    .value("ROM_FE", AtariRomFormat::ROM_FE)
    .value("ROM_F8", AtariRomFormat::ROM_F8)
    .value("ROM_F6", AtariRomFormat::ROM_F6)
    .value("ROM_NOT_SUPPORTED", AtariRomFormat::ROM_NOT_SUPPORTED)
    .export_values()
    ;

    py::class_<AtariRom>(m, "AtariRom")
    .def(py::init<const std::string&>())
    .def("reset", &AtariRom::reset)
    .def("minimal_actions", &AtariRom::minimal_actions)
    .def("file_name", &AtariRom::file_name)
    .def("game_name", &AtariRom::game_name)
    .def("type_name", &AtariRom::type_name)
    .def("md5", &AtariRom::md5)
    .def("swap_ports", &AtariRom::swap_ports)
    .def("use_paddles", &AtariRom::use_paddles)
    .def("swap_paddles", &AtariRom::swap_paddles)
    .def("allow_hmove_blanks", &AtariRom::allow_hmove_blanks)
    .def("player_left_difficulty_B", &AtariRom::player_left_difficulty_B)
    .def("player_right_difficulty_B", &AtariRom::player_right_difficulty_B)
    .def("is_supported", &AtariRom::is_supported)
    .def("is_ntsc", &AtariRom::is_ntsc)
    .def("ram_size", &AtariRom::ram_size)
    .def("rom_size", &AtariRom::rom_size)
    .def("screen_height", &AtariRom::screen_height)
    .def("screen_width", &AtariRom::screen_width)
    .def("screen_size", &AtariRom::screen_size)
    .def("type", &AtariRom::type)
    .def("has_banks", &AtariRom::has_banks)
    .def("game_id", &AtariRom::game_id)
    .def("data", &AtariRom::data)
    ;

    py::class_<AtariState>(m, "AtariState")
    .def(py::init())
    .def_readwrite("left_paddle", &AtariState::left_paddle)
    .def_readwrite("right_paddle", &AtariState::right_paddle)
    .def_readwrite("frame_number", &AtariState::frame_number)
    .def_readwrite("episode_frame_number", &AtariState::episode_frame_number)
    .def_readwrite("string_length", &AtariState::string_length)
    .def_readwrite("save_system", &AtariState::save_system)
    .def_readwrite("md5", &AtariState::md5)
    .def_readwrite("cycles", &AtariState::cycles)
    .def_readwrite("A", &AtariState::A)
    .def_readwrite("X", &AtariState::X)
    .def_readwrite("Y", &AtariState::Y)
    .def_readwrite("SP", &AtariState::SP)
    .def_readwrite("PC", &AtariState::PC)
    .def_readwrite("IR", &AtariState::IR)
    .def_readwrite("N", &AtariState::N)
    .def_readwrite("V", &AtariState::V)
    .def_readwrite("B", &AtariState::B)
    .def_readwrite("D", &AtariState::D)
    .def_readwrite("I", &AtariState::I)
    .def_readwrite("notZ", &AtariState::notZ)
    .def_readwrite("C", &AtariState::C)
    .def_readwrite("executionStatus", &AtariState::executionStatus)
    .def_readwrite("ram", &AtariState::ram)
    .def_readwrite("timer", &AtariState::timer)
    .def_readwrite("intervalShift", &AtariState::intervalShift)
    .def_readwrite("cyclesWhenTimerSet", &AtariState::cyclesWhenTimerSet)
    .def_readwrite("cyclesWhenInterruptReset", &AtariState::cyclesWhenInterruptReset)
    .def_readwrite("timerReadAfterInterrupt", &AtariState::timerReadAfterInterrupt)
    .def_readwrite("DDRA", &AtariState::DDRA)
    .def_readwrite("DDRB", &AtariState::DDRB)
    .def_readwrite("clockWhenFrameStarted", &AtariState::clockWhenFrameStarted)
    .def_readwrite("clockStartDisplay", &AtariState::clockStartDisplay)
    .def_readwrite("clockStopDisplay", &AtariState::clockStopDisplay)
    .def_readwrite("clockAtLastUpdate", &AtariState::clockAtLastUpdate)
    .def_readwrite("clocksToEndOfScanLine", &AtariState::clocksToEndOfScanLine)
    .def_readwrite("scanlineCountForLastFrame", &AtariState::scanlineCountForLastFrame)
    .def_readwrite("currentScanline", &AtariState::currentScanline)
    .def_readwrite("VSYNCFinishClock", &AtariState::VSYNCFinishClock)
    .def_readwrite("enabledObjects", &AtariState::enabledObjects)
    .def_readwrite("VSYNC", &AtariState::VSYNC)
    .def_readwrite("VBLANK", &AtariState::VBLANK)
    .def_readwrite("NUSIZ0", &AtariState::NUSIZ0)
    .def_readwrite("NUSIZ1", &AtariState::NUSIZ1)
    .def_readwrite("COLUP0", &AtariState::COLUP0)
    .def_readwrite("COLUP1", &AtariState::COLUP1)
    .def_readwrite("COLUPF", &AtariState::COLUPF)
    .def_readwrite("COLUBK", &AtariState::COLUBK)
    .def_readwrite("CTRLPF", &AtariState::CTRLPF)
    .def_readwrite("playfieldPriorityAndScore", &AtariState::playfieldPriorityAndScore)
    .def_readwrite("REFP0", &AtariState::REFP0)
    .def_readwrite("REFP1", &AtariState::REFP1)
    .def_readwrite("PF", &AtariState::PF)
    .def_readwrite("GRP0", &AtariState::GRP0)
    .def_readwrite("GRP1", &AtariState::GRP1)
    .def_readwrite("DGRP0", &AtariState::DGRP0)
    .def_readwrite("DGRP1", &AtariState::DGRP1)
    .def_readwrite("ENAM0", &AtariState::ENAM0)
    .def_readwrite("ENAM1", &AtariState::ENAM1)
    .def_readwrite("ENABL", &AtariState::ENABL)
    .def_readwrite("DENABL", &AtariState::DENABL)
    .def_readwrite("HMP0", &AtariState::HMP0)
    .def_readwrite("HMP1", &AtariState::HMP1)
    .def_readwrite("HMM0", &AtariState::HMM0)
    .def_readwrite("HMM1", &AtariState::HMM1)
    .def_readwrite("HMBL", &AtariState::HMBL)
    .def_readwrite("VDELP0", &AtariState::VDELP0)
    .def_readwrite("VDELP1", &AtariState::VDELP1)
    .def_readwrite("VDELBL", &AtariState::VDELBL)
    .def_readwrite("RESMP0", &AtariState::RESMP0)
    .def_readwrite("RESMP1", &AtariState::RESMP1)
    .def_readwrite("collision", &AtariState::collision)
    .def_readwrite("POSP0", &AtariState::POSP0)
    .def_readwrite("POSP1", &AtariState::POSP1)
    .def_readwrite("POSM0", &AtariState::POSM0)
    .def_readwrite("POSM1", &AtariState::POSM1)
    .def_readwrite("POSBL", &AtariState::POSBL)
    .def_readwrite("currentGRP0", &AtariState::currentGRP0)
    .def_readwrite("currentGRP1", &AtariState::currentGRP1)
    .def_readwrite("HMOVEBlankEnabled", &AtariState::HMOVEBlankEnabled)
    .def_readwrite("lastHMOVEClock", &AtariState::lastHMOVEClock)
    .def_readwrite("M0CosmicArkMotionEnabled", &AtariState::M0CosmicArkMotionEnabled)
    .def_readwrite("M0CosmicArkCounter", &AtariState::M0CosmicArkCounter)
    .def_readwrite("dumpEnabled", &AtariState::dumpEnabled)
    .def_readwrite("dumpDisabledCycle", &AtariState::dumpDisabledCycle)
    .def_readwrite("bank", &AtariState::bank)
    .def_readwrite("reward", &AtariState::reward)
    .def_readwrite("score", &AtariState::score)
    .def_readwrite("terminal", &AtariState::terminal)
    .def_readwrite("started", &AtariState::started)
    .def_readwrite("lives", &AtariState::lives)
    .def_readwrite("points", &AtariState::points)
    .def_readwrite("last_lives", &AtariState::last_lives)
    ;

    py::class_<AtariEnv>(m, "AtariEnv")
    .def(py::init<const AtariRom&, const size_t, const size_t>())
    .def("initialize", [](AtariEnv &env,
                          uint64_t statesBuffer,
                          uint64_t frameStatesBuffer,
                          uint64_t ramBuffer,
                          uint64_t tiaUpdateBuffer,
                          uint64_t frameBuffer,
                          uint64_t romIndicesBuffer,
                          uint64_t minimalActionsBuffer,
                          uint64_t randStatesBuffer,
                          uint64_t cachedStatesBuffer,
                          uint64_t cachedRamBuffer,
                          uint64_t cachedFrameStatesBuffer,
                          uint64_t cachedTiaUpdateBuffer,
                          uint64_t cacheIndexBuffer)
        {
            env.initialize_ptrs(reinterpret_cast<cule::atari::state*>(statesBuffer),
                                reinterpret_cast<cule::atari::frame_state*>(frameStatesBuffer),
                                reinterpret_cast<uint8_t*>(ramBuffer),
                                reinterpret_cast<uint32_t*>(tiaUpdateBuffer),
                                reinterpret_cast<uint8_t*>(frameBuffer),
                                reinterpret_cast<uint32_t*>(romIndicesBuffer),
                                reinterpret_cast<AtariAction*>(minimalActionsBuffer),
                                reinterpret_cast<uint32_t*>(randStatesBuffer),
                                reinterpret_cast<cule::atari::state*>(cachedStatesBuffer),
                                reinterpret_cast<uint8_t*>(cachedRamBuffer),
                                reinterpret_cast<cule::atari::frame_state*>(cachedFrameStatesBuffer),
                                reinterpret_cast<uint32_t*>(cachedTiaUpdateBuffer),
                                reinterpret_cast<uint32_t*>(cacheIndexBuffer));
        }
    )
    .def("reset", [](AtariEnv &env, uint64_t seedBuffer)
        {
            env.reset((uint32_t*) seedBuffer);
        }
    )
    .def("reset_states", [](AtariEnv &env)
        {
            env.reset_states();
        }
    )
    .def("step", [](AtariEnv &env, const bool fire_reset, uint64_t actionBuffer, uint64_t doneBuffer)
        {
            env.step(fire_reset,
                     reinterpret_cast<AtariAction*>(actionBuffer),
                     reinterpret_cast<uint8_t*>(doneBuffer));
        }
    )
    .def("two_step", [](AtariEnv &env, uint64_t playerABuffer, uint64_t playerBBuffer)
        {
            env.two_step((const AtariAction*) playerABuffer,
                         (const AtariAction*) playerBBuffer);
        }
    )
    .def("get_data", [](AtariEnv& env, const bool episodic_life, uint64_t doneBuffer, uint64_t rewardBuffer, uint64_t livesBuffer)
        {
            env.get_data(episodic_life,
                         reinterpret_cast<uint8_t*>(doneBuffer),
                         reinterpret_cast<int32_t*>(rewardBuffer),
                         reinterpret_cast<int32_t*>(livesBuffer));
        }
    )
    .def("generate_frames", [](AtariEnv& env, const bool rescale, const bool last_frame, const size_t num_channels, uint64_t imageBuffer)
        {
            env.generate_frames(rescale,
                                last_frame,
                                num_channels,
                                reinterpret_cast<uint8_t*>(imageBuffer));
        }
    )
    .def("generate_random_actions", [](AtariEnv& env, uint64_t actionBuffer)
        {
            env.generate_random_actions((AtariAction*) actionBuffer);
        }
    )
    .def("sync_other_stream", [](AtariEnv& env)
        {
            // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            // env.sync_other_stream(stream);
        }
    )
    .def("sync_this_stream", [](AtariEnv& env)
        {
            // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            // env.sync_this_stream(stream);
        }
    )
    .def("get_states", [](AtariEnv& env, const std::vector<int32_t>& indices)
        {
            std::vector<AtariState> atari_states(indices.size());
            env.get_states(indices.size(), indices.data(), atari_states.data());
            return atari_states;
        }
    )
    .def("set_states", [](AtariEnv& env, const std::vector<int32_t>& indices, const std::vector<AtariState>& atari_states)
        {
            env.set_states(indices.size(), indices.data(), atari_states.data());
        }
    )
    .def("set_cuda", &AtariEnv::set_cuda)
    .def("size", &AtariEnv::size)
    .def("image_buffer_size", &AtariEnv::image_buffer_size)
    .def("state_size", &AtariEnv::state_size)
    .def("frame_state_size", &AtariEnv::frame_state_size)
    .def("tia_update_size", &AtariEnv::tia_update_size)
    ;
}

