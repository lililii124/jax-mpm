"""NumPy reference for separate physical-gradient geometric transport."""
import numpy as np


def geometric_kinematics(F,L,increment_gradient,dt):
    eye=np.eye(3,dtype=np.asarray(F).dtype)
    star=L+increment_gradient
    A=eye+dt*star
    right_solve=lambda numerator,denominator:np.swapaxes(np.linalg.solve(
        np.swapaxes(denominator,-1,-2),np.swapaxes(numerator,-1,-2)),-1,-2)
    following=right_solve(star,A)
    midpoint=right_solve(star,eye+.5*dt*star)
    spin=.5*(midpoint-np.swapaxes(midpoint,-1,-2))
    rate=.5*(midpoint+np.swapaxes(midpoint,-1,-2))
    rotation=np.linalg.solve(eye-.5*dt*spin,eye+.5*dt*spin)
    half_rotation=np.linalg.solve(eye-.25*dt*spin,eye+.25*dt*spin)
    strain=dt*np.swapaxes(half_rotation,-1,-2)@rate@half_rotation
    return A@F,following,rotation,strain
