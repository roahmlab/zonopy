Math Operations
===============
.. automodule:: zonopy
    :members:
    :show-inheritance:
    :noindex:
.. currentmodule:: zonopy

Most common math operations are implemented for zonotopes and polynomial zonotopes.
The operations are implemented in a way that they can be used with both single and batched zonotopes.
In a later version, these operations will migrate to a dispatch mechanism to allow for more flexibility and better documentation.

Linear Algebra
--------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    cross

Transcendentals
---------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    cos
    sin
    cos_sin_cartprod

Rotations
---------
.. autosummary::
    :toctree: generated
    :nosignatures:

    rot_mat

Deprecated / Possibly Broken
----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    dot
    close
    gen_batch_rotatotope_from_jrs_trig
    gen_rotatotope_from_jrs_trig
    get_pz_rotations_from_q
