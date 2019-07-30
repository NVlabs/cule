#pragma once

#include <cule/config.hpp>

#include <cule/atari/games.hpp>

#define ALE_CASE(ID_NAME, NS_NAME, FUNC) \
case GAME_##ID_NAME:                            \
{                                               \
    FUNC(NS_NAME)                               \
    break;                                      \
}

#define GAME_SWITCH(FUNC)                                      \
switch(game_id)                                                \
{                                                              \
    ALE_CASE(ADVENTURE, adventure, FUNC)                \
    ALE_CASE(AIR_RAID, airraid, FUNC)                   \
    ALE_CASE(ALIEN, alien, FUNC)                        \
    ALE_CASE(AMIDAR, amidar, FUNC)                      \
    ALE_CASE(ASSAULT, assault, FUNC)                    \
    ALE_CASE(ASTERIX, asterix, FUNC)                    \
    ALE_CASE(ASTEROIDS, asteroids, FUNC)                \
    ALE_CASE(ATLANTIS, atlantis, FUNC)                  \
    ALE_CASE(BANK_HEIST, bankheist, FUNC)               \
    ALE_CASE(BATTLE_ZONE, battlezone, FUNC)             \
    ALE_CASE(BEAM_RIDER, beamrider, FUNC)               \
    ALE_CASE(BERZERK, berzerk, FUNC)                    \
    ALE_CASE(BOWLING, bowling, FUNC)                    \
    ALE_CASE(BOXING, boxing, FUNC)                      \
    ALE_CASE(BREAKOUT, breakout, FUNC)                  \
    ALE_CASE(CARNIVAL, carnival, FUNC)                  \
    ALE_CASE(CENTIPEDE, centipede, FUNC)                \
    ALE_CASE(CHOPPER, chopper, FUNC)                    \
    ALE_CASE(CRAZY_CLIMBER, crazyclimber, FUNC)         \
    ALE_CASE(DEFENDER, defender, FUNC)                  \
    ALE_CASE(DEMON_ATTACK, demonattack, FUNC)           \
    ALE_CASE(DOUBLE_DUNK, doubledunk, FUNC)             \
    ALE_CASE(ELEVATOR_ACTION, elevatoraction, FUNC)     \
    ALE_CASE(ENDURO, enduro, FUNC)                      \
    ALE_CASE(FISHING_DERBY, fishingderby, FUNC)         \
    ALE_CASE(FREEWAY, freeway, FUNC)                    \
    ALE_CASE(FROSTBITE, frostbite, FUNC)                \
    ALE_CASE(GOPHER, gopher, FUNC)                      \
    ALE_CASE(GRAVITAR, gravitar, FUNC)                  \
    ALE_CASE(HERO, hero, FUNC)                          \
    ALE_CASE(ICE_HOCKEY, icehockey, FUNC)               \
    ALE_CASE(JAMESBOND, jamesbond, FUNC)                \
    ALE_CASE(JOURNEY_ESCAPE, journeyescape, FUNC)       \
    ALE_CASE(KABOOM, kaboom, FUNC)                      \
    ALE_CASE(KANGAROO, kangaroo, FUNC)                  \
    ALE_CASE(KRULL, krull, FUNC)                        \
    ALE_CASE(KUNG_FU_MASTER, kungfumaster, FUNC)        \
    ALE_CASE(MONTEZUMA_REVENGE, montezumarevenge, FUNC) \
    ALE_CASE(MS_PACMAN, mspacman, FUNC)                 \
    ALE_CASE(NAME_THIS_GAME, namethisgame, FUNC)        \
    ALE_CASE(PHOENIX, phoenix, FUNC)                    \
    ALE_CASE(PINBALL, pinball, FUNC)                    \
    ALE_CASE(PITFALL, pitfall, FUNC)                    \
    ALE_CASE(PONG, pong, FUNC)                          \
    ALE_CASE(POOYAN, pooyan, FUNC)                      \
    ALE_CASE(PRIVATE_EYE, privateeye, FUNC)             \
    ALE_CASE(QBERT, qbert, FUNC)                        \
    ALE_CASE(RIVERRAID, riverraid, FUNC)                \
    ALE_CASE(ROAD_RUNNER, roadrunner, FUNC)             \
    ALE_CASE(ROBOTANK, robotank, FUNC)                  \
    ALE_CASE(SEAQUEST, seaquest, FUNC)                  \
    ALE_CASE(SKIING, skiing, FUNC)                      \
    ALE_CASE(SOLARIS, solaris, FUNC)                    \
    ALE_CASE(SPACE_INVADERS, spaceinvaders, FUNC)       \
    ALE_CASE(STAR_GUNNER, stargunner, FUNC)             \
    ALE_CASE(TENNIS, tennis, FUNC)                      \
    ALE_CASE(TIME_PILOT, timepilot, FUNC)               \
    ALE_CASE(TUTANKHAM, tutankham, FUNC)                \
    ALE_CASE(UP_N_DOWN, upndown, FUNC)                  \
    ALE_CASE(VENTURE, venture, FUNC)                    \
    ALE_CASE(WIZARD, wizard, FUNC)                      \
    ALE_CASE(YARS_REVENGE, yarsrevenge, FUNC)           \
    ALE_CASE(ZAXXON, zaxxon, FUNC)                      \
}

#define SET_TERMINAL(NAME) NAME::setTerminal(s);
#define IS_MINIMAL(NAME) value = NAME::isMinimal(a);
#define GET_REWARD(NAME) value = NAME::reward(s);
#define GET_SCORE(NAME) value = NAME::score(s);
#define GET_LIVES(NAME) value = NAME::lives(s);

namespace cule
{
namespace atari
{

struct ale
{

template<typename State_t>
static
CULE_ANNOTATION
void set_id(State_t& s, const games::GAME_TYPE& game_id)
{
    UPDATE_FIELD(s.riotData, FIELD_RIOT_GAME, game_id);
}

template<typename State_t>
static
CULE_ANNOTATION
games::GAME_TYPE get_id(State_t&s)
{
    return games::GAME_TYPE(SELECT_FIELD(s.riotData, FIELD_RIOT_GAME));
}

template<typename State_t>
static
CULE_ANNOTATION
void setTerminal(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    GAME_SWITCH(SET_TERMINAL)
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getRewards(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_REWARD)

    return value;
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getScore(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_SCORE)

    return value;
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getLives(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_LIVES)

    return value;
}

// is end of game
template<typename State_t>
static
CULE_ANNOTATION
bool isTerminal(State_t& s)
{
    return s.tiaFlags[FLAG_ALE_TERMINAL];
}

template<typename State_t>
static
CULE_ANNOTATION
bool isStarted(State_t& s)
{
    return s.tiaFlags[FLAG_ALE_STARTED];
}

template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.tiaFlags.clear(FLAG_ALE_STARTED);

    s.score = 0;
}

/** Drops illegal actions, such as the fire button in skiing. Note that this is different
  *   from the minimal set of actions. */
template<typename State_t>
static
CULE_ANNOTATION
void noopIllegalActions(State_t& s)
{
    s.sysFlags.clear(FLAG_CON_RESET);
}

}; // end namespace ale

CULE_ANNOTATION
bool isMinimal(const games::GAME_TYPE& game_id, const Action& a)
{
    using namespace games;

    bool value = false;
    GAME_SWITCH(IS_MINIMAL)

    return value;
}

// Returns the vector of the minimal set of actions needed to play
// the game.
std::vector<Action> getMinimalActionSet(const games::GAME_TYPE& game_id)
{
    std::vector<Action> actions;
    for (int i = 0; i < _ACTION_MAX; i++)
    {
        if(isMinimal(game_id, allActions[i]))
        {
            actions.push_back(allActions[i]);
        }
    }
    return actions;
}

} // end namespace atari
} // end namespace cule

