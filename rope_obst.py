from gym.envs.mujoco.dynamic_mjc.model_builder import MJCModel
import numpy as np
import os

np.random.seed(0)

COLORS = ["0.5 0.5 0.5 1", 
          "0.0 0.0 0.0 1",
          "0.0 0.5 0.0 1",
          "0.5 0.0 0.5 1",
          "0.0 0.0 0.5 1"]

def rope(num_beads = 5, 
    init_pos=[0.0, 0.0, 0.0],
    texture=False,
    ):
    mjcmodel = MJCModel('rope')
    mjcmodel.root.compiler(inertiafromgeom="auto",
        angle="radian",
        coordinate="local", 
        eulerseq="XYZ", 
        texturedir=os.path.dirname(os.path.realpath(__file__)) + "/../assets/textures")

    mjcmodel.root.size(njmax=6000, nconmax=6000)
    mjcmodel.root.option(timestep="0.005", gravity="0 0 -9.81", iterations="50", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(limited="false", damping="1")
    default.geom(contype="1", conaffinity="1", condim="3", friction=".5 .1 .1", density="1000", margin="0.002")

    worldbody = mjcmodel.root.worldbody()

    worldbody.camera(name="maincam", mode="fixed", fovy="32", euler="0.7 0 0", pos="0 -1.1 1.3")
    worldbody.camera(name="leftcam", mode="fixed", fovy="32", euler="0.7 0 -1.57", pos="-1.1 0 1.3")
    worldbody.camera(name="overheadcam", mode= "fixed", pos="0. 0. 1.5", euler="0.0 0.0 0.0")

    gripper = worldbody.body(name="gripper", pos=[0,0,0.25])

    gripper.inertial(pos="0 0 0", mass="1", diaginertia="16.667 16.667 16.667")
    gripper.geom(type="box", size=".1 .03 .03", rgba="0.5 0.5 0.5 0", contype="0", conaffinity="0")
    gripper.geom(type="box", size=".07 .035 1", rgba="0.9 0.9 0.9 0", pos="0 0 1", contype="0", conaffinity="0")

    gripper.joint(name="slide_x", type="slide", pos="0 0 0", axis="1 0 0", limited="true", range="-0.5 0.5", armature="0", damping="30", stiffness="0")
    gripper.joint(name="slide_y", type="slide", pos="0 0 0", axis="0 1 0", limited="true", range="-0.5 0.5", armature="0", damping="30", stiffness="0")
    gripper.joint(name="slide_z", type="slide", pos="0 0 0", axis="0 0 1",  limited="true", range="-0.08 0.15")
    gripper.joint(name="hinge_z", type="hinge", pos="0 0 0", axis="0 0 1", limited="true", range="-6.28 6.28", damping="30")

    fingers = gripper.body(name="fingers", pos=[0,0,0])
    finger_1 = fingers.body(name="finger_1", pos=[-0.08,0.0,-0.1])
    finger_1.joint(name="j_finger1", type="slide", pos="0 0 0", axis="1 0 0",  limited="true", range="0.0 0.0615")
    finger_1.geom(condim="6", contype="2", conaffinity="2", type="box", size=".01 .02 .1", rgba="0.3 0.3 0.3 0",  mass="0.08")
    finger_1.site(name="finger1_surf", pos="0.01 0 0", size=".0025 .0190 .095", type="box", rgba="0.0 1.0 0.0 0")

    finger_2 = fingers.body(name="finger_2", pos=[0.08,0.0,-0.1])
    finger_2.joint(name="j_finger2", type="slide", pos="0 0 0", axis="1 0 0")
    finger_2.geom(condim="6", contype="4", conaffinity="4",  type="box", size=".01 .02 .1", rgba="0.3 0.3 0.3 0", mass="0.08")
    finger_2.site(name="finger2_surf", pos="-0.01 0 0", size=".0025 .0190 .095", type="box", rgba="1.0 0.0 0.0 0")

    
    displacement = [0.0, 0.001, 0.0]
    
    #bead_pos = [0.0, 0.0, 0.015] #for cuboidal beads
    site_pos = [0.0, 0.0, 0.0] #for spherical beads
    tendon_range = [0.0, 0.01]

    color = np.random.choice(COLORS)
    # color = "0.5 0.5 0.5 1"

    beads = []
    for i in range(num_beads):
        new_pos = list(np.asarray(init_pos) + i*(np.asarray(displacement)))
        beads.append(worldbody.body(name="bead_{}".format(i), pos=new_pos))
        beads[i].joint(type="free")
        if texture:
            beads[i].geom(type="sphere", size="0.01", rgba=color,
                  mass="0.01", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
                  condim="6", solimp="0.99 0.99 0.01", solref="0.01 1", material="bead_material")
            # beads[i].geom(type="sphere", size="0.03", rgba="0.5 0.5 0.5 1", 
            #       mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
            #       condim="6", solimp="0.99 0.99 0.01", solref="0.01 1", material="bead_material")
        else:
            beads[i].geom(type="sphere", size="0.015", rgba="0 1 0 1",
             mass="0.01", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
             condim="6", solimp="0.99 0.99 0.01", solref="0.01 1")

        beads[i].site(name="site_{}".format(i), pos=site_pos, type="sphere", size="0.004")

        # beads[i].geom(type="box", size="0.015 0.03 0.015", rgba="0.8 0.2 0.2 1", 
        #           mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
        #           condim="6", solimp="0.99 0.99 0.01", solref="0.01 1")
        # beads[i].site(name="site_{}".format(i), pos=site_pos, type="sphere", size="0.01")

    container = worldbody.body(name="container", pos=[0,0,-0.05])
    #border_front = container.body(name="border_front", pos="0 -.5  0")
    #border_front.geom(type="box", size=".5 .01 .1", rgba="0 0 0 .3")
    #border_rear = container.body(name="border_rear", pos="0 .5  0")
    #border_rear.geom(type="box", size=".5 .01 .1", rgba="0 0 0 .3")
    #border_right = container.body(name="border_right", pos=".5 0. 0")
    #border_right.geom(type="box", size=".01  .5 .1", rgba="0 0 0 .3")
    #border_left = container.body(name="border_left", pos="-.5 0. 0")
    #border_left.geom(type="box", size=".01  .5 .1", rgba="0 0 0 .3")
    table = container.body(name="table", pos="0 0 -.01")
    if texture:
        table.geom(type="box", size=".5  .5 .01", rgba=".5 .5 .5 1", contype="7", conaffinity="7", material="table_material")
    else:
        table.geom(type="box", size=".5  .5 .01", rgba="0 0 0 1", contype="7", conaffinity="7")
        
    #light = worldbody.body(name="light", pos=[0,0,1])
    #light.light(name="light0", mode="fixed", directional="false", active="true", castshadow="true")

    tendons = mjcmodel.root.tendon()
    tendon_list = []
    for i in range(num_beads-1):
        tendon_list.append(tendons.spatial(limited="true", range=tendon_range, width="0.004"))
        tendon_list[i].site(site="site_{}".format(i))
        tendon_list[i].site(site="site_{}".format(i+1))

    actuator = mjcmodel.root.actuator()
    actuator.position(joint="slide_x", ctrllimited="false", kp="200")
    actuator.position(joint="slide_y", ctrllimited="false", kp="200")
    actuator.general(joint="slide_z", gaintype="fixed", dyntype="none", dynprm="1 0 0",
                    gainprm ="100 0 0", biastype="affine", biasprm="10 -100 -4")
    actuator.position(joint="hinge_z", ctrllimited="false", kp="300")
    actuator.motor(joint="j_finger1", ctrllimited="true", ctrlrange="-10.0 10.0")

    equality = mjcmodel.root.equality()
    equality.joint(joint1="j_finger1", joint2="j_finger2", polycoef="0 -1 0 0 0")

    asset = mjcmodel.root.asset()
    asset.texture(file='wood.png', name='table_texture')
    asset.material(name='table_material', rgba='1 1 1 1', shininess='0.3', specular='1', texture='table_texture')
    asset.texture(file='marble.png', name='bead_texture')
    asset.material(name='bead_material', rgba='1 1 1 1', shininess='0.3', specular='1', texture='bead_texture')

    obstacle = worldbody.body(name="obstacle", pos=[0, 0, 0])
    obstacle.geom(type="box", size=".03 .1 .03", rgba="1 0 0 1", contype="7", conaffinity="7")

    return mjcmodel

# <asset>
# <texture file="describable/dtd/images/waffled/waffled_0169.png" name="vase_texture" />
# <material name="vase_material" rgba="1 1 1 1" shininess="0.3" specular="1" texture="vase_texture" />
# </asset>
