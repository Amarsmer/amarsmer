import numpy as np

def S(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

def hydrodynamic(rg = np.zeros((3,1)), 
                 rb = np.zeros((3,1)), 
                 pose = np.zeros((3,1)), 
                 nu = np.zeros((6,1)), 
                 nudot = np.zeros((6,1)), 
                 added_masses = np.zeros((6,1)), 
                 viscous_drag = np.zeros((6,1)), 
                 quadratic_drag = np.zeros((6,1)), 
                 inertia=None):
    # --- Physical constants ---
    rho = 1026  # Density of water (kg/m^3)
    G = 9.81    # Gravitational acceleration (m/s^2)

    # --- Robot parameters ---
    R_r = 0.1     # Radius (m)
    L_r = 0.6     # Length (m)
    m_r = 13      # Mass (kg)
    V_r = np.pi * R_r**2 * L_r  # Volume of a cylinder (m^3)

    # Center of gravity and buoyancy
    '''
    rg = np.zeros((3,1))
    rb = np.zeros((3,1))
    '''
    xg, yg, zg = rg
    xb, yb, zb = rb

    # Orientation angles
    #angles = np.array([0.0, 0.0, 0.0])
    #angles = pose[3:]
    phi, theta, psi = pose[3:]

    # Linear and angular velocities
    #nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu_1 = nu[:3]
    nu_2 = nu[3:]

    # Accelerations
    #nudot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nudot_1 = nudot[:3]
    nudot_2 = nudot[3:]

    # --- Drag coefficients ---
    #viscous = -np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    Xu, Yv, Zw, Kp, Mq, Nr = viscous_drag

    #quadratic = -np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    Xuu, Yvv, Zww, Kpp, Mqq, Nrr = quadratic_drag

    # --- Inertia matrices ---
    I_cm = np.diag([
        0.5 * m_r * R_r**2,
        (1/12)*m_r*L_r**2 + 0.25*m_r*R_r**2,
        (1/12)*m_r*L_r**2 + 0.25*m_r*R_r**2
    ])

    I_parr = np.array([
        [yg**2 + zg**2, -xg * yg, -xg * zg],
        [-xg * yg, xg**2 + zg**2, -yg * zg],
        [-xg * zg, -yg * zg, xg**2 + yg**2]
    ])

    I_b = I_cm + m_r * I_parr

    # --- M matrix ---
    M_rb = np.block([
        [m_r * np.eye(3), -m_r * S(rg)],
        [m_r * S(rg), I_b]
    ])

    M_a = -np.diag(added_masses)

    M = M_rb + M_a

    # --- C matrix ---
    C_rb = np.block([
        [m_r * S(nu_2), -m_r * S(nu_2) @ S(rg)],
        [m_r * S(rg) @ S(nu_2), S(nu_2) @ I_b]
    ])

    C_a = np.block([[np.zeros((3,3)), S(np.multiply(added_masses[:3],nu1))],
                    [S(np.multiply(added_masses[:3],nu1)), S(np.multiply(added_masses[3:],nu2))]])

    C = C_rb + C_a

    # --- G vector ---
    W = m_r * G
    B = rho * G * V_r

    G = np.array([
        (W - B) * np.sin(theta),
        -(W - B) * np.cos(theta) * np.sin(phi),
        -(W - B) * np.cos(theta) * np.cos(phi),
        -(yg * W - zb * B) * np.cos(theta) * np.cos(phi) + (zg * W - zb * B) * np.cos(theta) * np.sin(phi),
        (zg * W - zb * B) * np.sin(theta) + (xg * W - xb * B) * np.cos(theta) * np.cos(phi),
        -(xg * W - xb * B) * np.cos(theta) * np.sin(phi) - (yg * W - yb * B) * np.sin(theta)
    ])

    # --- D matrix ---
    D = -np.diag([
        Xu + Xuu * abs(nu[0]),
        Yv + Yvv * abs(nu[1]),
        Zw + Zww * abs(nu[2]),
        Kp + Kpp * abs(nu[3]),
        Mq + Mqq * abs(nu[4]),
        Nr + Nrr * abs(nu[5])
    ])

    # --- Final tau computation ---
    tau = M @ nudot + C @ nu + G + D @ nu
