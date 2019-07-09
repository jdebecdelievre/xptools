from itertools import cycle

# Column names
pos_raw = ['Position' + x for x in ['X', 'Y', 'Z']]
quat_raw = ['Rotation' + x for x in ['W','X', 'Y', 'Z']]
pos = ['x_e', 'y_e', 'z_e']
quat = ['q0','qx', 'qy', 'qz']
eul = ['phi', 'theta', 'psi']
eul_deg = ['phi_deg', 'theta_deg', 'psi_deg']
pos_eul = pos + eul
rot = ['p','q','r']
vel = ['u', 'v', 'w']
frc = ['fx','fy','fz']
mom = ['mx','my','mz']
time = ['time']
mrp = ['sx', 'sy', 'sz'] # modified rodriguez parameters
raw2clean = dict(zip(pos_raw+quat_raw+['Time'],pos+quat+time))

# Derivative
dpos = [p+'_dot' for p in pos]
deul = [p+'_dot' for p in eul]
dquat = [p+'_dot' for p in quat]
drot = [p+'_dot' for p in rot]
dvel = [p+'_dot' for p in vel]

# Second Derivative
ddpos = [p+'_dot_dot' for p in pos]
ddeul = [p+'_dot_dot' for p in eul]
ddquat = [p+'_dot_dot' for p in quat]
ddrot = [p+'_dot_dot' for p in rot]
ddvel = [p+'_dot_dot' for p in vel]

# Filtered
fpos = [p+'_filt' for p in pos]
feul = [p+'_filt' for p in eul]
fquat =[p+'_filt' for p in quat]
frot = [p+'_filt' for p in rot]
fvel = [p+'_filt' for p in vel]

# Aero force and moments
aerof = ['D','Y','L']
aerocf = ['CD','CY','CL']
aerocm = ['Cl','Cm','Cn']

# performance variables
perf = ['L_D']


col_names=pos+eul+rot+vel+quat+dpos+deul+drot+dvel+dquat+frc+mom+aerof+aerocf+aerocm+perf

# Associated title name
col_titles = dict(zip(pos, ['X-position in Earth fr (m)', 'Y-position in Earth fr (m)', 'Z-position in Earth fr (m)']))
col_titles.update(dict(zip(dpos,['X-velocity in Earth fr (m/s)', 'Y-velocity in Earth fr (m/s)', 'Z-velocity in Earth fr (m/s)'])))
col_titles.update(dict(zip(vel,[v+ ' (m/s)' for v in vel])))
col_titles.update(dict(zip(rot,[r + ' (rad/s)' for r in rot])))
col_titles.update(dict(zip(eul,[e + ' (rad)' for e in eul])))
col_titles.update(dict(zip(eul_deg,[e + ' (deg)' for e in eul])))
col_titles.update(dict(zip(dvel,dvel)))
col_titles.update(dict(zip(drot,drot)))
col_titles.update(dict(zip(deul,deul)))
col_titles.update(dict(zip(quat,['q0', 'qx', 'qy', 'qz'])))
col_titles.update(dict(zip(dquat,['q0_dot', 'qx_dot', 'qy_dot', 'qz_dot'])))
col_titles.update(dict(zip(frc,frc)))
col_titles.update(dict(zip(mom,mom)))
col_titles.update(dict(zip(aerof,aerof)))
col_titles.update(dict(zip(aerocf,aerocf)))
col_titles.update(dict(zip(aerocm,aerocm)))
col_titles.update(dict(zip(perf, ['L/D'])))
col_titles.update(dict(zip(mrp, ['$\sigma_x$', '$\sigma_y$', '$\sigma_z$'])))


# Associated css colors
colors = [
(255,0,0),
(255,255,0),
(0,234,255),
(170,0,255),
(255,127,0),
(191,255,0),
(0,149,255),
(255,0,170),
(255,212,0),
(106,255,0),
(0,64,255),
(237,185,185),
(185,215,237),
(231,233,185),
(220,185,237),
(185,237,224),
(143,35,35),
(35,98,143),
(143,106,35),
(107,35,143),
(79,143,35),
(0,0,0),
(115,115,115),
(204,204,204)]

for c in range(len(colors)) :
	colors[c] = [x/255 for x in colors[c]]
ite = cycle(colors)

col_colors = {}
for c in col_names:
	col_colors[c] = next(ite)
