"""Pick-and-place on four robot arms in PyBullet (self-written for Experiment 8).

Embodiments: kuka_iiwa (7 DOF), franka_panda (7 DOF), xarm6 (6 DOF) and
kuka_mini (the iiwa loaded with globalScaling = 0.6, so same joints, shorter links).

Common interface (the "task space"): an action is an end-effector position
change (dx, dy, dz) in metres, at most STEP per axis, plus a gripper command
(+1 close, -1 open). Each arm turns the target into joint angles with its own
inverse kinematics. The gripper is abstracted: "close" attaches the cube when
the end effector is within GRASP_DIST of it, "open" releases it and lets it
fall under gravity. Robot-cube collisions are disabled so that no arm-specific
finger geometry is involved.
"""
import numpy as np
import pybullet as p
import pybullet_data

STEP = 0.02            # max end-effector move per action (m)
GRASP_DIST = 0.03      # attach radius (m)
SUCCESS_DIST = 0.03    # cube-goal distance counted as success (m)
CUBE_Z = 0.025         # cube centre when resting on the floor (5 cm cube)
APPROACH = 0.12        # approach height above the cube (m)
MAX_STEPS = 150

ROBOTS = {
    "kuka_iiwa":    dict(urdf="kuka_iiwa/model.urdf", scale=1.0, ee=6, joints=list(range(7)),
                         rest=[0, 0.4, 0, -1.4, 0, 1.2, 0]),
    "franka_panda": dict(urdf="franka_panda/panda.urdf", scale=1.0, ee=11, joints=list(range(7)),
                         rest=[0, -0.3, 0, -2.2, 0, 2.0, 0.8]),
    "xarm6":        dict(urdf="xarm/xarm6_robot.urdf", scale=1.0, ee=6, joints=[1, 2, 3, 4, 5, 6],
                         rest=[0, 0.2, -1.0, 0, 0.8, 0]),
    "kuka_mini":    dict(urdf="kuka_iiwa/model.urdf", scale=0.6, ee=6, joints=list(range(7)),
                         rest=[0, 0.4, 0, -1.4, 0, 1.2, 0]),
}
NAMES = list(ROBOTS)

# task region (m): cube on the left, goal on the right, in front of the base
CUBE_X, CUBE_Y = (0.35, 0.55), (0.05, 0.20)
GOAL_X, GOAL_Y = (0.35, 0.55), (-0.20, -0.05)


def sample_task(rng):
    cube = np.array([rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y), CUBE_Z])
    goal = np.array([rng.uniform(*GOAL_X), rng.uniform(*GOAL_Y), CUBE_Z])
    return cube, goal


class Arm:
    """One robot in its own DIRECT physics client."""

    def __init__(self, name, scale=None, rest_pose=True):
        cfg = ROBOTS[name]
        self.name, self.cfg = name, cfg
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        s = cfg["scale"] if scale is None else scale
        self.robot = p.loadURDF(cfg["urdf"], [0, 0, 0], useFixedBase=True, globalScaling=s,
                                physicsClientId=self.cid)
        self.cube = p.loadURDF("cube_small.urdf", [0.5, 0.1, CUBE_Z], physicsClientId=self.cid)
        for l in range(-1, p.getNumJoints(self.robot, physicsClientId=self.cid)):
            p.setCollisionFilterPair(self.robot, self.cube, l, -1, 0, physicsClientId=self.cid)
        self.joints, self.ee = cfg["joints"], cfg["ee"]
        info = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in self.joints]
        self.lo = np.array([i[8] for i in info]); self.hi = np.array([i[9] for i in info])
        self.rest = np.array(cfg["rest"], float)
        self.use_rest = rest_pose
        self.dof = len(self.joints)
        # IK returns one value per movable joint of the URDF
        self.movable = [j for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid))
                        if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] != p.JOINT_FIXED]

    # ---------------- kinematics ----------------
    def set_q(self, q):
        for j, v in zip(self.joints, q):
            p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)

    def q(self):
        return np.array([p.getJointState(self.robot, j, physicsClientId=self.cid)[0] for j in self.joints])

    def ee_pos(self):
        return np.array(p.getLinkState(self.robot, self.ee, computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def ik(self, target, iters=3):
        """Position-only IK. For the 7-DOF arms the rest pose fills the free (null-space) DOF."""
        kw = {}
        if self.use_rest:
            n = len(self.movable)
            lo = np.full(n, -3.0); hi = np.full(n, 3.0); rest = np.zeros(n)
            for k, j in enumerate(self.movable):
                if j in self.joints:
                    i = self.joints.index(j); lo[k], hi[k], rest[k] = self.lo[i], self.hi[i], self.rest[i]
            kw = dict(lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                      jointRanges=(hi - lo).tolist(), restPoses=rest.tolist())
        q = self.q()
        for _ in range(iters):                   # a few refinements from the current pose
            sol = np.array(p.calculateInverseKinematics(
                self.robot, self.ee, list(map(float, target)), maxNumIterations=100,
                residualThreshold=1e-4, physicsClientId=self.cid, **kw))
            q = np.clip([sol[self.movable.index(j)] for j in self.joints], self.lo, self.hi)
            self.set_q(q)
        return q

    # ---------------- episode ----------------
    def reset(self, cube, goal):
        self.set_q(self.rest)
        self.goal = np.array(goal, float)
        p.resetBasePositionAndOrientation(self.cube, list(map(float, cube)), [0, 0, 0, 1],
                                          physicsClientId=self.cid)
        self.held, self.t = False, 0
        return self.obs()

    def cube_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)[0])

    def obs(self):
        return dict(ee=self.ee_pos(), cube=self.cube_pos(), goal=self.goal.copy(),
                    held=float(self.held), q=self.q())

    def step(self, action):
        """action = (dx, dy, dz, grip). Returns (obs, timeout)."""
        d = np.clip(np.asarray(action[:3], float), -STEP, STEP)
        target = self.ee_pos() + d
        target[2] = max(target[2], 0.01)
        self.ik(target)
        self.apply_grip(action[3])
        self.t += 1
        return self.obs(), self.t >= MAX_STEPS

    def apply_grip(self, g):
        ee = self.ee_pos()
        if g > 0 and not self.held and np.linalg.norm(ee - self.cube_pos()) < GRASP_DIST:
            self.held = True
        if self.held:
            if g < 0:
                self.held = False
                for _ in range(120):            # let the cube drop and settle
                    p.stepSimulation(physicsClientId=self.cid)
            else:
                p.resetBasePositionAndOrientation(self.cube, ee.tolist(), [0, 0, 0, 1], physicsClientId=self.cid)
                p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def goal_error(self):
        c = self.cube_pos()
        return float(np.linalg.norm(c[:2] - self.goal[:2])), bool(c[2] < 0.05)

    def close(self):
        p.disconnect(self.cid)


class Expert:
    """Scripted task-space expert: above cube, down, close, lift, above goal, down, open."""

    def __init__(self):
        self.phase = 0
        self.lift_xy = None

    def act(self, o):
        ee, cube, goal = o["ee"], o["cube"], o["goal"]
        lift = CUBE_Z + APPROACH
        if self.phase == 2:                                   # close the gripper
            self.phase = 3
            return np.array([0, 0, 0, 1.0]), False
        if self.phase in (3, 4, 5) and not o["held"]:         # grasp missed or cube lost: retry
            self.phase, self.lift_xy = 0, None
        if self.phase == 6:                                   # open: episode ends after this
            return np.array([0, 0, 0, -1.0]), True
        wp = {0: cube + [0, 0, APPROACH], 1: cube,
              3: np.r_[self.lift_xy if self.lift_xy is not None else ee[:2], lift],
              4: np.r_[goal[:2], lift], 5: goal + [0, 0, 0.01]}[self.phase]
        delta = wp - ee
        if np.linalg.norm(delta) < 0.01:
            if self.phase == 1:
                self.lift_xy = ee[:2].copy()
            self.phase += 1
            return self.act(o)
        grip = 1.0 if self.phase in (3, 4, 5) else -1.0
        return np.r_[np.clip(delta, -STEP, STEP), grip], False


def run_expert(arm, cube, goal, noise=0.0, rng=None, record=False):
    """Expert episode. With noise > 0 the executed motion is perturbed (DART); labels stay clean."""
    o, ex, log = arm.reset(cube, goal), Expert(), []
    path, finished = [o["ee"]], False
    while True:
        a, finished = ex.act(o)
        if record:
            log.append((o, a.copy()))
        ex_a = a.copy()
        if noise > 0 and not finished:
            ex_a[:3] += rng.normal(0, noise, 3)
        o, timeout = arm.step(ex_a)
        path.append(o["ee"])
        if finished or timeout:
            break
    err, on_floor = arm.goal_error()
    ok = bool(finished and on_floor and err < SUCCESS_DIST)
    return dict(success=ok, steps=arm.t, err=err, path=np.array(path), log=log)


def run_policy(arm, policy, cube, goal):
    """Roll out a learned policy; the episode ends when it opens the gripper while holding, or times out."""
    o = arm.reset(cube, goal)
    path, released = [o["ee"]], False
    while True:
        a = policy(o)
        was_held = arm.held
        o, timeout = arm.step(a)
        path.append(o["ee"])
        if was_held and not arm.held:
            released = True
            break
        if timeout:
            break
    err, on_floor = arm.goal_error()
    return dict(success=bool(released and on_floor and err < SUCCESS_DIST), steps=arm.t, err=err,
                path=np.array(path))
