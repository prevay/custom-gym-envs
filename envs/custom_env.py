
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import itertools
from operator import attrgetter

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._xy_bounds = [range(0,5), range(0,5)]
        self._previous_loc = None
        self._start = [0, 0]
        self._goal = [4, 3]
        self._holes = [[1, 1], [3, 2]]
        self._loc = [0, 0]
        self._t = 0
        self._max_t = 8

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0, 0, 0]), np.array([4, 4, 8]))


    def step(self, action):
        #Actions: 0=U, 1=R, 2=D, 3=L

        self._previous_loc = self._loc[:]
        self._t +=1

        if action == 0:
            self._loc[1] +=1
        if action == 1:
            self._loc[0] +=1
        if action == 2:
            self._loc[1] -=1
        if action == 3:
            self._loc[0] -= 1

        reward = 0
        done = False

        dist = abs(self._loc[0] - self._goal[0]) + abs(self._loc[1] - self._goal[1])
        prev_dist = abs(self._previous_loc[0] - self._goal[0]) + \
                    abs(self._previous_loc[1] - self._goal[1])
        if dist < prev_dist:
            reward = 0.05
        else:
            reward = -0.1

        if self._loc == self._goal:
            reward = 1
            done = True
        if self._loc in self._holes or self._loc[0] not in self._xy_bounds[0] or \
                self._loc[1] not in self._xy_bounds[1] or self._t > self._max_t:
            reward = -1
            done = True

        return [*self._loc, self._t], reward, done, {}


    def reset(self):
        self._loc = self._start[:]
        self._t = 0
        self._previous_loc = None
        return [*self._loc, self._t]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class BasicGridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._xy_bounds = [range(0,5), range(0,5)]
        self._start = [0, 0]
        self._goal = [4, 3]
        self._loc = [0, 0]
        self._t = 0
        self._max_t = 8
        self._previous_loc = None
        self._holes = [[3, 0], [2, 3]]
        # self.reward_range = [-0.1, 1]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0, 0, 0]), np.array([4, 4, 8]))

    def step(self, action):
        #Actions: 0=U, 1=R, 2=D, 3=L

        self._t +=1
        self._previous_loc = self._loc[:]
        new_x = self._loc[0]
        new_y = self._loc[1]

        if action == 0:
            new_y = self._loc[1] + 1
        if action == 1:
            new_x = self._loc[0] + 1
        if action == 2:
            new_y = self._loc[1] - 1
        if action == 3:
            new_x = self._loc[0] - 1

        reward = 0
        done = False

        #if new_x in self._xy_bounds[0] and new_y in self._xy_bounds[1]:
        self._loc = [new_x, new_y]

        dist = abs(self._loc[0] - self._goal[0]) + abs(self._loc[1] - self._goal[1])
        prev_dist = abs(self._previous_loc[0] - self._goal[0]) + \
                    abs(self._previous_loc[1] - self._goal[1])
        if dist < prev_dist:
            reward = 0.05
        if dist > prev_dist:
            reward = -0.1

        if self._loc == self._goal:
            reward = 1
            done = True
        elif self._t > self._max_t or self._loc[0] not in self._xy_bounds[0] \
                or self._loc[1] not in self._xy_bounds[1] or self._loc in self._holes:
            done = True

        return [*self._loc, self._t], reward, done, {}

    def reset(self):
        self._loc = self._start[:]
        self._t = 0
        self._previous_loc = None
        return [*self._loc, self._t]

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class ForageWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._xy_bounds = [range(0,7), range(0,7)]
        self._loc = None
        self._hp = 10
        self._t = 0
        self._max_t = 100
        self._plants = {(3,2) : 5, (1,5): 5, (4, 6): 5, (5, 2): 5}
        self._mushrooms = {(3,4) : -5, (3,5): -5, (5, 4): -5}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(9,))

    def step(self, action):
        #Actions: 0=U, 1=R, 2=D, 3=L

        self._t +=1
        reward = 1
        done = False
        self._hp -= 1

        self.move(action)
        self.forage()

        if self._hp <= 0 or self._t == self._max_t \
                or self._loc[0] not in self._xy_bounds[0]\
                or self._loc[1] not in self._xy_bounds[1]:
            reward = 0
            done = True

        self.step_vegetation()
        current_nbhd = self.get_nbhd()

        return current_nbhd, reward, done, {}

    def reset(self):
        self._loc = list(np.random.randint(7, size=2)) # hard-coded for now
        self._t = 0
        self._hp = 10
        return self.get_nbhd()

    def move(self, action):
        if action == 0:
            self._loc[1] += 1
        if action == 1:
            self._loc[0] += 1
        if action == 2:
            self._loc[1] -= 1
        if action == 3:
            self._loc[0] -= 1

    def forage(self):
        if tuple(self._loc) in self._plants.keys():
            self._hp  += min(10 - self._hp, self._plants[tuple(self._loc)])
            self._plants[tuple(self._loc)] = 0
        elif tuple(self._loc) in self._mushrooms.keys():
            self._hp += min(self._hp, self._mushrooms[tuple(self._loc)])
            # self._mushrooms[tuple(self._loc)] = 0

    def get_nbhd(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                out_of_bounds = -10 if any(x < 0 or x > 6 for x in loc) else 0
                nbhd.append(self._plants.get(loc, 0) +
                            self._mushrooms.get(loc, 0) +
                            out_of_bounds)
        return nbhd

    def step_vegetation(self):
        for plant, val in self._plants.items():
            self._plants[plant] = min(5, val + 0.5)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class NPCAgent():

    # TODO: make sure agents don't act when hp < 0
    def __init__(self, init_loc_range, max_hp):
        self._hp = max_hp
        self._max_hp = max_hp
        self._init_loc_range = init_loc_range
        self._loc = [np.random.randint(list(init_loc_range[0])[-1]),
                     np.random.randint(list(init_loc_range[1])[-1])]

    @property
    def loc(self):
        return self._loc

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, val):
        self._hp = val
        
    def reset(self):
        self._hp = self._max_hp
        self._loc = [np.random.randint(list(self._init_loc_range[0])[-1]),
                     np.random.randint(list(self._init_loc_range[1])[-1])]

    def move(self, loc_range):
        new_x = -1
        new_y = -1
        while new_x not in list(loc_range[0]) or new_y not in list(loc_range[1]):
            d = np.random.randint(low=-1, high=2, size=2)
            new_x = self._loc[0] + d[0]
            new_y = self._loc[1] + d[1]
        self._loc = [new_x, new_y]
        self._hp -= 1


class Foe(NPCAgent):
    def __init__(self, init_loc_range, max_hp):
        super().__init__(init_loc_range, max_hp)

    def forage(self, plants):
        if tuple(self._loc) in plants.keys():
            self._hp += min(10 - self._hp, plants[tuple(self._loc)])
            plants[tuple(self._loc)] = 0


class Friend(NPCAgent):
    def __init__(self, init_loc_range, max_hp):
        super().__init__(init_loc_range, max_hp)

    def weed_out_mushrooms(self, shrooms):
        if tuple(self._loc) in shrooms.keys():
            shrooms[tuple(self._loc)] = 0


class HuntersAndGatherers(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._xy_bounds = [range(0,7), range(0,7)]
        self._loc = None
        self._hp = 10
        self._t = 0
        self._max_t = 100
        self._plants = self.init_plants(3)
        self._mushrooms = self.init_mushrooms(3)
        self._foes = [Foe(self._xy_bounds, 10)]
        self._friends = [Friend(self._xy_bounds, 20)]
        self.vegetation_t_minus1 = None
        self.vegetation_t_minus2 = None
        self.vegetation_t = None

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple([spaces.Box(low=-10, high=5, shape=(9,)),
                                               spaces.Box(low=-10, high=5, shape=(9,)),
                                               spaces.Box(low=-10, high=5, shape=(9,)),
                                              spaces.Box(low=0, high=1, shape=(9,)),
                                              spaces.Box(low=0, high=1, shape=(9,))])

    def init_plants(self, n):
        plant_dict = {}
        placed = 0
        while placed < n:
            x = np.random.randint(low=list(self._xy_bounds[0])[0],
                                  high=list(self._xy_bounds[0])[-1])
            y = np.random.randint(low=list(self._xy_bounds[1])[0],
                                  high=list(self._xy_bounds[1])[-1])
            if (x, y) not in plant_dict.keys():
                plant_dict[(x, y)] = 5
                placed += 1

        return plant_dict

    def init_mushrooms(self, n):
        shroom_dict = {}
        placed = 0
        while placed < n:
            x = np.random.randint(low=list(self._xy_bounds[0])[0],
                                  high=list(self._xy_bounds[0])[-1])
            y = np.random.randint(low=list(self._xy_bounds[1])[0],
                                  high=list(self._xy_bounds[1])[-1])
            if (x, y) not in shroom_dict.keys() and (x, y) not in self._plants:
                shroom_dict[(x, y)] = -5
                placed += 1

        return shroom_dict

    def step(self, action):
        #Actions: 0=U, 1=R, 2=D, 3=L, 4=Attack, 5=Share

        self._t +=1
        reward = 1
        done = False
        self._hp -= 1

        self.move(action)
        self.interact(action)
        self.forage()
        # have foe and friend act:
        for foe in self._foes:
            if foe.hp > 0:
                foe.move(self._xy_bounds)
                foe.forage(self._plants) 
        for friend in self._friends:
            if friend.hp > 0:
                friend.move(self._xy_bounds)
                friend.weed_out_mushrooms(self._mushrooms)
        if np.random.random() < 0.1 :
            self.spawn_mushroom()

        if self._hp <= 0 or self._t == self._max_t \
                or self._loc[0] not in self._xy_bounds[0]\
                or self._loc[1] not in self._xy_bounds[1]:
            reward = 0
            done = True

        self.step_vegetation()
        self.vegetation_t_minus2 = self.vegetation_t_minus1
        self.vegetation_t_minus1 = self.vegetation_t
        self.vegetation_t = self.get_vegetation_view()
        foes = self.get_foe_view()
        friends = self.get_friend_view()

        return (self.vegetation_t, 
                self.vegetation_t_minus1, 
                self.vegetation_t_minus2, 
                foes, 
                friends),\
               reward, done, {}

    def reset(self):
        # TODO: reset foes and friends (hp, loc, etc.)
        self._loc = list(np.random.randint(7, size=2)) # hard-coded for now
        self._t = 0
        self._hp = 10
        for foe in self._foes:
            foe.reset()
        for friend in self._friends:
            friend.reset()
        n = np.random.randint(2,9)
        self._plants = self.init_plants(n)
        self._mushrooms = self.init_mushrooms(n)
        self.vegetation_t_minus1 = self.get_vegetation_view()
        self.vegetation_t_minus2 = self.get_vegetation_view()
        self.vegetation_t = self.get_vegetation_view()
        return (self.vegetation_t,
                self.vegetation_t_minus1,
                self.vegetation_t_minus2,
                self.get_foe_view(),
                self.get_friend_view())

    def move(self, action):
        if action == 0:
            self._loc[1] += 1
        if action == 1:
            self._loc[0] += 1
        if action == 2:
            self._loc[1] -= 1
        if action == 3:
            self._loc[0] -= 1

    def interact(self, action):
        if action == 4: # attack neighboring foe
            foe = None
            for dy in [- 1, 0, 1]:
                for dx in [-1, 0, 1]:
                    loc = [self._loc[0] + dx, self._loc[1] + dy]
                    try:
                        foe = next(f for f in self._foes if f.loc == loc)
                        break
                    except:
                        continue
            if foe:
                # the location does not seem to align with the foe's location
                self._loc = loc
                damage = self._hp - foe.hp
                if damage > 0:
                    foe.hp -= damage
                else:
                    self._hp += damage
        if action == 5:
            friend = None
            for dy in [- 1, 0, 1]:
                for dx in [-1, 0, 1]:
                    loc = [self._loc[0] + dx, self._loc[1] + dy]
                    try:
                        friend = next(f for f in self._friends if f.loc == loc)
                        break
                    except:
                        continue
            if friend:
                self._loc = loc
                friend.hp += self._hp / 2
                self._hp = self._hp / 2

    def forage(self):
        if tuple(self._loc) in self._plants.keys():
            self._hp  += min(10 - self._hp, self._plants[tuple(self._loc)])
            self._plants[tuple(self._loc)] = 0
        elif tuple(self._loc) in self._mushrooms.keys():
            self._hp += min(self._hp, self._mushrooms[tuple(self._loc)])

    def spawn_mushroom(self):
        x = np.random.randint(low=list(self._xy_bounds[0])[0],
                              high=list(self._xy_bounds[0])[-1])
        y = np.random.randint(low=list(self._xy_bounds[1])[0],
                              high=list(self._xy_bounds[1])[-1])
        self._mushrooms[(x,y)] = -5

    def get_vegetation_view(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                out_of_bounds = -10 if any(x < 0 or x > 6 for x in loc) else 0
                nbhd.append(self._plants.get(loc, 0) +
                            self._mushrooms.get(loc, 0) +
                            out_of_bounds)
        return nbhd

    def get_foe_view(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                val = 0
                for foe in self._foes:
                    if foe.loc == loc:
                        val += 1
                nbhd.append(val)
        return nbhd

    def get_friend_view(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                val = 0
                for friend in self._friends:
                    if friend.loc == loc:
                        val += 1
                nbhd.append(val)
        return nbhd

    def step_vegetation(self):
        plants_updated = {}
        for plant, val in self._plants.items():
            self._plants[plant] = min(5, val + 0.5)
            if np.random.random() < 0.1:
                x = np.random.randint(low=list(self._xy_bounds[0])[0],
                                      high=list(self._xy_bounds[0])[-1])
                y = np.random.randint(low=list(self._xy_bounds[1])[0],
                                      high=list(self._xy_bounds[1])[-1])
                while (x, y) in self._mushrooms.keys() or(x, y) in self._plants.keys():
                    x = np.random.randint(low=list(self._xy_bounds[0])[0],
                                          high=list(self._xy_bounds[0])[-1])
                    y = np.random.randint(low=list(self._xy_bounds[1])[0],
                                          high=list(self._xy_bounds[1])[-1])
                plants_updated[(x, y)] = 5
            else:
                plants_updated[plant] = self._plants[plant]

        self._plants = plants_updated

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class Other():

    # TODO: make sure agents don't act when hp < 0
    def __init__(self, max_hp, id=None):
        self._id = id
        self._hp = max_hp
        self._max_hp = max_hp
        self._loc = None

    @property
    def loc(self):
        return self._loc

    @property
    def hp(self):
        return self._hp

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @hp.setter
    def hp(self, val):
        self._hp = val

    @loc.setter
    def loc(self, val):
        self._loc = val

    def reset(self):
        self._hp = self._max_hp
        self._loc = None

    def forage(self, plants):
        self._hp -= 1
        if tuple(self._loc) in plants.keys():
            self._hp  += min(10 - self._hp, plants[tuple(self._loc)])
            plants[tuple(self._loc)] = 0
        return plants


class HuntersAndGatherersMultiPlayer(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._id = None
        self._xy_bounds = [range(0,7), range(0,7)]
        self._loc = None
        self._hp = 10
        self._t = 0
        self._plants = None
        self._others = []
        self.vegetation_t_minus1 = None
        self.vegetation_t_minus2 = None
        self.vegetation_t = None

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple([spaces.Box(low=-10, high=5, shape=(9,)),
                                               spaces.Box(low=-10, high=5, shape=(9,)),
                                               spaces.Box(low=-10, high=5, shape=(9,)),
                                              spaces.Box(low=0, high=1, shape=(9,))])

    @property
    def loc(self):
        return self._loc

    @property
    def t(self):
        return self._t

    @property
    def hp(self):
        return self._hp

    @property
    def others(self):
        return self._others

    @property
    def id(self):
        return self._id

    @hp.setter
    def hp(self, val):
        self._hp = val

    @others.setter
    def others(self, val):
        self._others = val

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def plants(self):
        return self._plants

    def step(self, actions):
        #Actions: 0=U, 1=R, 2=D, 3=L, 4=Attack, 5=Share

        self._t +=1
        reward = 1
        done = False
        self._hp -= 1


        # TODO: make sure it's kosher to do this before the agent actions are taken
        # self.step_vegetation()
        self._plants = actions[3]
        self._others = [other for other in self._others if other.hp > 0]

        # TODO: how to handle dead agents?
        self.move(actions)
        self.interact(actions)
        self.forage()

        if self._hp <= 0 \
                or self._loc[0] not in self._xy_bounds[0]\
                or self._loc[1] not in self._xy_bounds[1]:
            reward = 0
            done = True

        self.vegetation_t_minus2 = self.vegetation_t_minus1
        self.vegetation_t_minus1 = self.vegetation_t
        self.vegetation_t = self.get_vegetation_view()
        others = self.get_other_view()

        return (self.vegetation_t,
                self.vegetation_t_minus1,
                self.vegetation_t_minus2,
                others),\
               reward, done, {}

    def reset(self, plants=None, id=None):
        if not id:
            self._loc = list(np.random.randint(7, size=2)) # hard-coded for now
        else:
            original_other = Other(10, self._id)
            original_other.hp = self._hp
            original_other.loc = self._loc
            # I think the order here is no longer important:
            self._others = [*self._others[0:self._id], original_other, *self._others[self._id:]]
            self._id = id
        self._t = 0
        self._hp = 10
        if plants:
            self._plants = plants
        self.vegetation_t_minus1 = self.get_vegetation_view()
        self.vegetation_t_minus2 = self.get_vegetation_view()
        self.vegetation_t = self.get_vegetation_view()
        return (self.vegetation_t,
                self.vegetation_t_minus1,
                self.vegetation_t_minus2,
                self.get_other_view())

    def add_others(self, locs, ids):
        for loc, id in list(zip(locs, ids)):
            other = Other(10)
            other.loc = loc
            other.id = id
            self._others.append(other)

    def move(self, actions):
        actions = actions[0]
        if actions[0] == 0:
            self._loc[1] += 1
        if actions[0] == 1:
            self._loc[0] += 1
        if actions[0] == 2:
            self._loc[1] -= 1
        if actions[0] == 3:
            self._loc[0] -= 1
        for action, other in list(zip(actions[1:], self._others)):
            if other.hp > 0:
                if action == 0:
                    other.loc[1] += 1
                if action == 1:
                    other.loc[0] += 1
                if action == 2:
                    other.loc[1] -= 1
                if action == 3:
                    other.loc[0] -= 1
            if other.loc[0] not in self._xy_bounds[0]\
                or other.loc[1] not in self._xy_bounds[1]:
                other.hp = 0


    def interact(self, actions):
        permutations = actions[1]
        entity_perm = actions[2]
        actions = actions[0]
        other = None
        break_again = False
        for dy in list(itertools.permutations([- 1, 0, 1]))[permutations[0][0]]:
            for dx in list(itertools.permutations([- 1, 0, 1]))[permutations[0][1]]:
                loc = [self._loc[0] + dx, self._loc[1] + dy]
                try:
                    entity_list = sorted([*self._others, self], key=attrgetter('id'))
                    entity_perm = next(p for i, p in enumerate(itertools.permutations(entity_list))
                                       if i == entity_perm)
                    other = next(f for f in entity_perm if f.loc == loc and f.hp > 0 and f != self)
                    break_again = True
                    break
                except StopIteration:
                    continue
            if break_again:
                break
        if other and actions[0] == 4:
            self._loc = loc
            damage = self._hp - other.hp
            if damage > 0:
                other.hp -= damage
            else:
                self._hp += damage
        if other and actions[0] == 5:
            self._loc = loc
            other.hp += self._hp / 2
            self._hp = self._hp / 2
        break_again = False
        for action, other, perm in list(zip(actions[1:], self._others, permutations[1:])):
            alter = None
            if other.hp > 0:
                for dy in list(itertools.permutations([- 1, 0, 1]))[perm[0]]:
                    for dx in list(itertools.permutations([- 1, 0, 1]))[perm[1]]:
                        try:
                            loc = [other.loc[0] + dx, other.loc[1] + dy]
                        except TypeError:
                            print('error')
                        try:
                            entity_list = sorted([*self._others, self], key=attrgetter('id'))
                            entity_perm = next(p for i, p in enumerate(itertools.permutations(entity_list))
                                               if i == entity_perm)
                            alter = next(f for f in entity_perm if f.loc == loc and f != other and f.hp > 0)
                            break_again = True
                            break
                        except StopIteration:
                            continue
                    if break_again:
                        break
                if alter and action == 4:
                    # the location does not seem to align with the foe's location
                    other.loc = loc
                    damage = other.hp - alter.hp
                    if damage > 0:
                        alter.hp -= damage
                    else:
                        other.hp += damage
                if alter and action == 5:
                    other.loc = loc
                    alter.hp += other.hp / 2
                    other.hp = other.hp / 2

    def forage(self):
        if tuple(self._loc) in self._plants.keys():
            self._hp  += min(10 - self._hp, self._plants[tuple(self._loc)])
            self._plants[tuple(self._loc)] = 0
        for other in self._others:
            if other.hp > 0:
                self._plants = other.forage(self._plants)


    def get_vegetation_view(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                out_of_bounds = -10 if any(x < 0 or x > 6 for x in loc) else 0
                nbhd.append(self._plants.get(loc, 0) + out_of_bounds)
        return nbhd

    def get_other_view(self):
        nbhd = []
        for dy in [- 1, 0, 1]:
            for dx in [-1, 0 , 1]:
                loc = (self._loc[0] + dx, self._loc[1] + dy)
                val = 0
                for other in self._others:
                    if other.loc == loc and other.hp > 0:
                        val += 1
                nbhd.append(val)
        return nbhd

    def render(self, mode='human'):
        pass

    def close(self):
        pass

