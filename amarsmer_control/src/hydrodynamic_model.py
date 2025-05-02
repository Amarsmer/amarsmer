#!/usr/bin/env python3

import numpy as np

def S(vec):
    vec = np.asarray(vec).flatten()
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def hydrodynamic(rg = np.zeros(3), 
                 rb = np.zeros(3), 
                 eta = np.zeros(6), 
                 nu = np.zeros(6), 
                 nudot = np.zeros(6), 
                 added_masses = np.zeros(6), 
                 viscous_drag = np.zeros(6), 
                 quadratic_drag = np.zeros(6), 
                 inertia=None):
    # --- Flatten all input vectors ---
    rg = np.asarray(rg).flatten()
    rb = np.asarray(rb).flatten()
    eta = np.asarray(eta).flatten()
    nu = np.asarray(nu).flatten()
    nudot = np.asarray(nudot).flatten()
    added_masses = np.asarray(added_masses).flatten()
    viscous_drag = np.asarray(viscous_drag).flatten()
    quadratic_drag = np.asarray(quadratic_drag).flatten()

    # --- Physical constants ---
    rho = 1026  # Density of water (kg/m^3)
    G_acc = 9.81    # Gravitational acceleration (m/s^2)

    # --- Robot parameters ---
    R_r = 0.1     # Radius (m)
    L_r = 0.6     # Length (m)
    m_r = 13      # Mass (kg)
    V_r = np.pi * R_r**2 * L_r  # Volume of a cylinder (m^3)

    # Center of gravity and buoyancy
    xg, yg, zg = rg
    xb, yb, zb = rb

    # Orientation angles
    phi, theta, psi = eta[3:]

    # Linear and angular velocities
    nu_1 = nu[:3]
    nu_2 = nu[3:]

    # Accelerations
    nudot_1 = nudot[:3]
    nudot_2 = nudot[3:]

    # --- Drag coefficients ---
    Xu, Yv, Zw, Kp, Mq, Nr = viscous_drag
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

    am_linear = added_masses[:3]
    am_angular = added_masses[3:]

    C_a = np.block([
        [np.zeros((3,3)), S(am_linear * nu_1)],
        [S(am_linear * nu_1), S(am_angular * nu_2)]
    ])

    C = C_rb + C_a

    # --- G vector ---
    W = m_r * G_acc
    B = rho * G_acc * V_r

    G_vec = np.array([
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
    tau = M @ nudot + C @ nu + G_vec + D @ nu
    return tau
