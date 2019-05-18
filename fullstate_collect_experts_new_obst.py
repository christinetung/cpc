from scipy.misc import imshow, imsave
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from tqdm import tqdm

#from gym.envs.mujoco.rope_meta_classifier import RopeMetaClassifierEnv
#from cubic_spline import generate_points

from gym.envs.mujoco.rope_oracle import RopeOracleEnv
#from sac.envs.normalized_env import normalize

np.random.seed(0)
random.seed(0)

def check_boundaries(qpos, random_xy):
    num_beads = env.num_beads
    xy = []

    qpos = env.sim.data.qpos

    for i in range(num_beads):
        xy.append(qpos[init_joint_offset + i*num_free_joints:][:2])

    xy = np.asarray(xy)
    x_min = np.min(xy[:,0])
    y_min = np.min(xy[:,1])

    x_max = np.max(xy[:,0])
    y_max = np.max(xy[:,1])

    xy_min = np.asarray([x_min, y_min])
    xy_max = np.asarray([x_max, y_max])


    if (random_xy[0] + xy_min[0] > -1.0*x_lim and random_xy[1] + xy_min[1] > -1.0*y_lim and
        random_xy[0] + xy_max[0] < x_lim and random_xy[1] + xy_max[1] < y_lim):
        return True
    else:
        return False

def generate_task(env, qpos, qpos_noise, id):
    """
    env: env
    qpos: list of qpos to be applied to the env
    id: task id
    qpos_noise: was earlier being gor negatives, not being used right now
    """
    num_img = 0
    save_folder = os.path.join(save_dir, 'task_%d'%id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.savetxt(os.path.join(save_folder, 'qpos_original.txt'), env.sim.data.qpos)

    for i in range(len(qpos)):
        #print(qpos[i][5:8])
        env.set_state(qpos[i], env.init_qvel)
        img = env.sim.render(img_h, img_w, camera_name='overheadcam')
        imsave(os.path.join(save_folder, 'success_%d.png'%(num_img)), img)
        np.savetxt(os.path.join(save_folder, 'success_qpos_%d.txt'%(num_img)), qpos[i])

        #env.set_state(qpos_noise[i], env.init_qvel)

        # for _ in range(20):
        #     env.sim.step()
        
        # img = env.sim.render(img_h, img_w, camera_name='overheadcam')
        # imsave(os.path.join(save_folder, 'failure_%d.png'%(num_img)), img)

        num_img+=1


def generate_displacements(env, num_displacements_small, num_displacements_big):

    num_beads = env.num_beads
    xy = []

    qpos = env.sim.data.qpos

    for i in range(num_beads):
        xy.append(qpos[init_joint_offset + i*num_free_joints:][:2])

    xy = np.asarray(xy)
    x_min = np.min(xy[:,0])
    y_min = np.min(xy[:,1])

    x_max = np.max(xy[:,0])
    y_max = np.max(xy[:,1])

    xy_min = np.asarray([x_min, y_min])
    xy_max = np.asarray([x_max, y_max])

    random_xy_list = []

    for i in range(num_displacements_big):
        done=False
        while not done:
            random_xy = np.random.uniform(-1.0*big_noise, big_noise, 2)

            if (random_xy[0] + xy_min[0] > -1.0*x_lim and random_xy[1] + xy_min[1] > -1.0*y_lim and
                random_xy[0] + xy_max[0] < x_lim and random_xy[1] + xy_max[1] < y_lim):
                done=True
                random_xy_list.append(random_xy)

    for i in range(num_displacements_small):
        done=False
        while not done:
            random_xy = np.zeros((2,))

            if (random_xy[0] + xy_min[0] > -1.0*x_lim and random_xy[1] + xy_min[1] > -1.0*y_lim and
                random_xy[0] + xy_max[0] < x_lim and random_xy[1] + xy_max[1] < y_lim):
                done=True
                random_xy_list.append(random_xy)

    random.shuffle(random_xy_list)
    
    return random_xy_list


def generate_qpos(env, displacements):
    qpos_orig = env.sim.data.qpos
    num_beads = env.num_beads
    
    qpos_new_list = []
    qpos_new_list_mid_noise = []

    for i in range(len(displacements)):

        qpos_new = np.copy(qpos_orig)
        qpos_new_mid_noise = np.copy(qpos_orig)

        for j in range(num_beads):
            offset = init_joint_offset + j*num_free_joints
            qpos_new[offset:offset+2] = qpos_orig[offset:offset+2] \
                    + displacements[i] + np.random.uniform(-1.0*small_noise, small_noise, 2)
        
            qpos_new_mid_noise[offset:offset+2] = qpos_orig[offset:offset+2] \
                    + displacements[i] + np.random.choice(np.asarray([-1.0, +1.0]))*np.random.uniform(1.5*small_noise, mid_noise, 2)
        
        qpos_new_list_mid_noise.append(qpos_new_mid_noise)
        qpos_new_list.append(qpos_new)

    return qpos_new_list, qpos_new_list_mid_noise

'''if __name__ == "__main__":
    #env params
    texture = False
    num_beads = 25
    init_pos = [0.0, 0.0, 0.0]

    #env = normalize(RopeOracleEnv())
    env = RopeOracleEnv(texture=texture, num_beads=num_beads, init_pos=init_pos)

    #dataset params
    img_h = 256
    img_w = 256

    #multi task setup
    num_tasks = 2
    num_displacements_small = 50
    num_displacements_big = 150

    # single task setup
    # num_tasks = 10
    # num_displacements_small = 50
    # num_displacements_big = 1450

    disp_thresh = 5.0
    dataset_name = 1

    init_joint_offset = 6
    num_free_joints = 7
    small_noise = 0.01
    mid_noise = 0.05
    big_noise = 0.2
    x_lim = 0.35
    y_lim = 0.35

    #save_dir = 'data/data_rand_act_0'
    #save_dir = '/mount/outputs/logs/rope_data/data/data_rand_act_4'
    save_dir = 'data'

    #save_dir = 'data/data_individual_tasks'

    #neagtive_dir = os.path.join(save_dir, 'common_negatives')
    #if not os.path.exists(neagtive_dir):
    #    os.makedirs(neagtive_dir)

    #env = RopeMetaClassifierEnv()

    x_neutral = -0.5
    y_neutral = -0.5
    z_min = -0.1
    z_max = +0.05
    torque_max = +10.0
    torque_neutral = 0.0
    torque_min = -1*torque_max

    #action_neutral = np.asarray([x_neutral, y_neutral, z_max, 0.0, torque_max])
    #env.do_pos_simulation_with_substeps(action_neutral)
    #env.do_pos_simulation_with_substeps(action_neutral)
    img = env.sim.render(img_h, img_w, camera_name='overheadcam')

    #import IPython; IPython.embed()
    MAX_ACT = 8
    num_fail = 0
    i=0

    n = 0

    #for i in tqdm(range(num_tasks)):
    while i < num_tasks:
        print('Task %d' % i)
        env.reset()
        qpos_orig = np.copy(env.sim.data.qpos)
        max_disp = 0.0

        num_act = 0
        while max_disp < disp_thresh:

            #random_push = np.random.uniform(-1, +1, 4)
            random_push = np.random.normal(loc=0, scale=0.25, size=4)
            env.push(random_push)
            num_act+=1
            qpos_new = env.sim.data.qpos

            img = env.sim.render(img_h, img_w, camera_name='overheadcam')
            traj_dir = os.path.join(save_dir, str(num_fail))
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
            if n % 5 == 0:
                imsave(os.path.join(traj_dir, '%d.png' % n), img)
            n += 1

            if not check_boundaries(env, np.asarray([0.,0.])) or n >= 500:

                #img = env.sim.render(img_h, img_w, camera_name='overheadcam')
                #imsave(os.path.join(neagtive_dir, 'fail_%d.png'%(num_fail)), img)
                num_fail+=1
                n = 0
                init_pos = np.random.normal(loc=0, scale=5.0, size=3)
                print(init_pos)
                env.init_pos = init_pos
                env.reset()
                qpos_orig = np.copy(env.sim.data.qpos)
                max_disp = 0.0
                continue

            for j in range(num_beads):
                offset = init_joint_offset + j*num_free_joints
                disp = np.linalg.norm(qpos_orig[offset:offset+2] - qpos_new[offset:offset+2])
                if disp > max_disp:
                    max_disp = disp

            # if max_disp < disp_thresh:
            #     img = env.sim.render(img_h, img_w, camera_name='overheadcam')
            #     imsave(os.path.join(neagtive_dir, 'fail_%d.png'%(num_fail)), img)
            #     num_fail+=1

        #print(num_act)
        if num_act < MAX_ACT:
            displacements = generate_displacements(env, num_displacements_small, num_displacements_big)
            qpos, qpos_noise = generate_qpos(env, displacements)
            generate_task(env, qpos, qpos_noise, i)
            i+=1
        else:
            pass


        print('Tasks completeted: {}'.format(i))'''

#####

if __name__ == "__main__":
    texture = False
    num_beads = 10
    init_joint_offset = 6
    num_free_joints = 7
    x_lim = 0.4
    y_lim = 0.4

    num_trajs = 1000
    traj_length = 100
    k_steps = 1
    img_h = 128
    img_w = 128
    save_dir = 'obst'

    data = []
    for i in range(num_trajs):
        print('Trajectory: %d' % i)
        init_pos = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.05), 0.0]
        print('Initial position: ' + str(init_pos))
        env = RopeOracleEnv(texture=texture, num_beads=num_beads, init_pos=init_pos)
        env.reset()

        traj = []
        for n in range(traj_length * k_steps):
            #random_push = np.random.normal(loc=0, scale=0.4, size=4)
            b = np.random.randint(low=0, high=num_beads)
            offset = init_joint_offset + b * num_free_joints
            x = env.sim.data.qpos[offset:offset + 2]
            dx = np.random.normal(loc=0, scale=0.2, size=2)
            random_push = [x[0], x[1], x[0] + dx[0], x[1] + dx[1]]
            env.push(random_push)
            img = env.sim.render(img_h, img_w, camera_name='overheadcam')
            if i < 10:
                traj_dir = os.path.join(save_dir, 'traj_%d' % i)
                if not os.path.exists(traj_dir):
                    os.makedirs(traj_dir)
                imsave(os.path.join(traj_dir, '%d.png' % n), img)
            traj.append(img)
            if not check_boundaries(env, np.asarray([0., 0.])):
                break
        traj = np.array(traj)
        data.append(traj)
    data = np.array(data)
    np.save('data_obst_large_128.npy', data)
