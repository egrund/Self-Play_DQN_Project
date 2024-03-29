from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

#from ..base.errors import UnavailableActionError, EpisodeDoneError
# next three classes source: https://github.com/KristianHolsheimer/keras-gym/blob/12d83ec9de644cd25b0a7ed4fe250850f6b998ea/keras_gym/base/errors.py
class KerasGymError(Exception):
    pass

class EpisodeDoneError(KerasGymError):
    pass

class UnavailableActionError(KerasGymError):
    pass


__all__ = (
    'TikTakToeEnv',
)


class TikTakToeEnv(Env):
    """

    Attributes
    ----------
    action_space : gym.spaces.Discrete(9)
        The action space.

    observation_space : MultiDiscrete(nvec)

        **Note:** The "current" player is relative to whose turn it is, which
        means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap between
        turns.

    max_time_steps : int
        Maximum number of timesteps within each episode.

    available_actions : array of int
        Array of available actions. This list shrinks when columns saturate.

    win_reward : 1.0
        The reward associated with a win.

    loss_reward : -1.0
        The reward associated with a loss.

    draw_reward : 0.0
        The reward associated with a draw.
        
    move_reward : -0.1
        The reward associated with a move.

    wrong_move_reward : -0.5
        The reward associated with an unavailable move. 
    """  # noqa: E501
    # class attributes
    num_rows = 3
    num_cols = 3
    num_players = 2
    win_reward = 1.0
    loss_reward = -win_reward
    draw_reward = 0.0
    move_reward = 0# -0.1 #0
    wrong_move_reward = -0.5
    action_space = Discrete(num_cols * num_rows)
    observation_space = MultiDiscrete(
        nvec=np.full((num_rows, num_cols, num_players), 2, dtype='uint8'))
    max_time_steps = int(num_rows * num_cols)
    filters = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0]],
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]],
        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]],
        [[1, 0, 0],
         [1, 0, 0],
         [1, 0, 0]],
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]],
    ], dtype='uint8')

    def __init__(self):
        self._init_state()


    def reset(self):
        """
        Reset the environment to the starting position.

        Returns
        -------
        s : 3d-array, shape: [num_rows + 1, num_cols, num_players]

            A state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        """
        self._init_state()
        return self.state


    def step(self, a, return_wrong : bool = False):
        """
        Take one step in the MDP, following the single-player convention from
        gym.

        Parameters
        ----------
        a : int, options: {0, 1, 2, 3, 4, 5, 6, 7, 8}
            The action to be taken. The action is the zero-based count of the
            possible insertion slots, starting from the top left of the board.

        return_wrong: boolean
            True when we want to return whether or not the action was available and give a negativ reward for wrong actions
            False when we want to raise an error for unavailable actions

        Returns
        -------
        s_next : array, shape [6, 7, 2]

            A next-state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        r : float
            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

            **Note:** Since "current" player is relative to whose turn it is,
            you need to be careful about aligning the rewards with the correct
            state or state-action pair. In particular, this reward :math:`r` is
            the one associated with the :math:`s` and :math:`a`, i.e. *not*
            aligned with :math:`s_\\text{next}`.

        done : bool
            Whether the episode is done.

        wrong : bool
            Whether the action was an available action

        #info : dict or None
            #A dict with some extra information (or None).

        """
        if self.done:
            raise EpisodeDoneError("please reset env to start new episode")
        if not self.action_space.contains(a):
            raise ValueError("invalid action ", a)
        if a not in self.available_actions:
            if return_wrong:
                # the last return value now is the wrong bool
                return self.state, self.wrong_move_reward, self.done , True #, {'state_id': self.state_id}
            else:
                raise UnavailableActionError("action is not available")

        # swap players
        self._players = np.roll(self._players, -1)

        # update state
        self._state[a] = self._players[0]
        self._prev_action = a

        # run logic
        self.done, reward = self._done_reward(a)
        return (self.state, reward, self.done, False) if return_wrong else (self.state, reward, self.done) #, {'state_id': self.state_id}


    def render(self, *args, **kwargs):
        """
        Render the current state of the environment.

        """
        # lookup for symbols
        symbol = {
            1: u'\u25CF',   # player 1 token (agent)
            2: u'\u25CB',   # player 2 token (adversary)
            -1: u'\u25BD',  # indicator for player 1's last action
            -2: u'\u25BC',  # indicator for player 2's last action
        }

        # render board
        inbetween = '     '
        hrule = '+---' * self.num_cols + '+' + inbetween + '+---' * self.num_cols + '+\n'
        board = " Game                  Actions \n"
        #board += "   ".join(
            # symbol.get(-(a == self._prev_action) * self._players[1], " ")
            #symbol.get( not (a == self._prev_action) * self._players[1], " ")
            #for a in range(self.num_cols))
        #board += "  \n"
        board += hrule
        s = np.reshape(self._state,(self.num_rows, self.num_cols))
        a = np.reshape(np.linspace(0,self.action_space.n -1,self.action_space.n),(self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            board += "| "
            board += " | ".join(
                symbol.get(s[i, j], " ")
                for j in range(self.num_cols))
            board += " |" + inbetween + "| "
            board += " | ".join(
                str(int(a[i,j])) for j in range(self.num_cols)
            )
            board += " |\n"
            board += hrule
        #board += "  0   1   2   3   4   5   6  \n"  # actions

        print(board)


    @property
    def state(self):
        stacked_layers = np.stack((
            np.reshape((self._state == self._players[0]).astype('uint8'),(self.num_rows, self.num_cols)),
            np.reshape((self._state == self._players[1]).astype('uint8'),(self.num_rows, self.num_cols)),
        ), axis=-1)  # shape: [num_rows, num_cols, num_players]
        #available_actions_mask = np.zeros(
            #(1, self.num_cols, self.num_players), dtype='uint8')
        #available_actions_mask[0, self.available_actions, :] = 1
        #return np.concatenate((available_actions_mask, stacked_layers), axis=0)
        return stacked_layers

    """
    @property
    def state_id(self):
        p = str(self._players[0])
        d = '1' if self.done else '0'
        if self._prev_action is None:
            a = str(self.num_cols)
        else:
            a = str(self._prev_action)
        s = ''.join(self._state.ravel().astype('str'))  # base-3 string
        s = '{:017x}'.format(int(s, 3))  # 17-char hex string
        return p + d + a + s             # 20-char hex string

    def set_state(self, state_id):
        # decode state id
        p = int(state_id[0], 16)
        d = int(state_id[1], 16)
        a = int(state_id[2], 16)
        assert p in (1, 2)
        assert d in (0, 1)
        assert self.action_space.contains(a) or a == self.num_cols
        self._players[0] = p    # 1 or 2
        self._players[1] = 3 - p  # 2 or 1
        self.done = d == 1
        self._prev_action = None if a == self.num_cols else a
        s = np.base_repr(int(state_id[3:], 16), 3)
        z = np.zeros(self.num_rows * self.num_cols, dtype='uint8')
        z[-len(s):] = np.array(list(s), dtype='uint8')
        self._state = z.reshape((self.num_rows, self.num_cols))
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        for j in range(self.num_cols):
            for i in self._state[::-1, j]:
                if i == 0:
                    break
                self._levels[j] -= 1
    """

    @property
    def available_actions(self):
        actions = np.argwhere(self._state == 0).ravel()
        return actions

    @property
    def available_actions_mask(self): 
        mask = np.zeros(self.num_cols * self.num_rows, dtype='bool')
        mask[self.available_actions] = True
        return mask

    def _init_state(self):
        self._prev_action = None
        self._players = np.array([1, 2], dtype='uint8')
        self._state = np.zeros((self.num_rows * self.num_cols), dtype='uint8')
        #self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        self.done = False

    def _done_reward(self, a):
        """
        Check whether the last action `a` by the current player resulted in a
        win or draw for player 1 (the agent). This contains the main logic and
        implements the rules of the game.

        """
        assert self.action_space.contains(a)

        # update filling levels
        # self._levels[a] -= 1

        s = np.reshape(self._state == self._players[0],(self.num_rows, self.num_cols))
        #for i0 in range(1):
            #i1 = i0 + 3
            #for j0 in range(1):
                # j1 = j0 + 3
                #if np.any(np.tensordot(self.filters, s[i0:i1, j0:j1]) == 4):
                    #return True, 1.0

        if np.any(np.tensordot(self.filters, s) == 3):
            return True, self.win_reward

        # check for a draw
        if len(self.available_actions) == 0:
            return True, self.draw_reward

        # this is what's returned throughout the episode
        return False, self.move_reward
