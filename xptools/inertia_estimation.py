import json
from collections import namedtuple
import numpy as np
import click
from xptools.aircraft_geom_definition import AircraftGeom

def mass_cg_estimation(aircraft_file):
    # Load
    geom = AircraftGeom(aircraft_file)
    
    if geom.mass_measurements['front_scale'] is None:
        print(f'No mass_measurements field in {aircraft_file}. Skiping cg location estimation')
        return
    
    # Find CG location
    scales = geom.mass_measurements
    D = scales['distance_between_scales']
    d = scales['front_scale_to_nose']
    Mf = scales['front_scale']
    Mb = scales['back_scale']

    geom.mass_measured = Mf + Mb
    geom.CGlocation_measured = [d + D * Mb / (Mf + Mb), 0.0, 0.0]

    # Dump
    geom.dump()

# <h1> Estimating trapezoid inertia </h1>
# We look at a trapezoid with the following name convention
# <img src="trapezoid.gif">
# We estimate centroids and moments of inertia using the following formulas:
# http://www.efunda.com/math/areas/trapezoid.cfm
# Ixy is computed using a decomposition of the trapezoid in 2 rectangles and one rectangle.
# <img src="trap2.png">
# We can verify all calculations using this calculator: 
# https://calcresource.com/moment-of-inertia-trap.html
# 
# On a half wing we typically have:
# - b = root chord
# - a = tip chord
# - h = span
# - c = b/4 + h*tan(sweep) - a/4 (sweep is defined at quarter chord)
# 
# For a vertical tail, rotate axis around Ox by 90degrees
# 
# A few notes: 
# - parallel axis theorem only works to move to/from centroid
# - Ixy sign flips if we symmetrize the geometry


trap = namedtuple('trap',['A','Ix','Iy', 'Iz', 'Ixz', 'Ixc','Iyc', 'Izc', 'Ixzc'])
def trapezoid(h,a,b,c):
   # centroid
    yc = h/3 * (2*a + b)/ (a+b)
    xc = (2*a*c + a**2 + c*b + a*b +b**2)/(3*(a+b))
    
    # area
    A = h*(a+b)/2
    
    # moment inertia around x
    Ix = h**3 * (3*a + b)/12 # along root
    Ixc = h**3 * (a**2 + 4*a*b + b**2)/36/(a+b) # through centroid
    
    # moment inertia around y
    Iy = h/12 * (a**3 + 3*a*c**2 + 3*a**2*c + b**3 + c*b**2 +                  a*b**2 + b*c**2 + 2*a*b*c + b*a**2)# along leading edge
    Iyc = h * (4*a*b*c**2 + 3*a**2*b*c - 3*a*b**2*c + a**4 + b**4 +               2*a**3*b + a**2*c**2 + a**3*c + 2*a*b**3 -                c*b**3 + b**2*c**2)/36/(a+b)# through centroid
    
    # moment inertia around z
    Iz = h/12*(h**2*b + 3*h**2*a + a**3 + 3*a**2*c + b**3 +               b**2*c + a*b**2 + b*c**2 + 2*a*b*c + a**2*b)# at leading edge
    Izc = h * (4*h**2*a*b + h**2*b**2 + h**2*a**2 + 4*a*b*c**2 + 3*a**2*b*c              -3*a*b**2*c + a**4 + b**4 + 2*a**3*b + a**2*c**2 + a**3*c +               2*a*b**3 - b**3*c + b**2*c**2)/36/(a+b) # through centroid
    
    # product of inertia around xy
    Ixy = h**2 /2 * (a*(c+a/2) - (b-c-a)**2/36 + c**2/36)         + 1/9*c**2*h**2 + 1/18 * h**2 * (b-c-a) * (b + 2*c + 2*a) # bottom left corner / leading edge fuselage point
    Ixyc = Ixy - A*yc*xc # through centroid
    return xc, yc, A, Ix, Ixc, Iy, Iyc, Iz, Izc, Ixy, Ixyc


def structure_inertia(geom):
    inertias = namedtuple('inertias',field_names=['Ixx','Iyy', 'Izz', 'Ixz','A'])
    for k, val in zip(geom._fields, geom):
        print('{}:{}'.format(k, val))

    CG_est = 0.0

    # # Wing
    D = geom.Wing

    # Estimate moments of area for a half wing at its centroid
    h = D['span']/2
    a = D['tip chord']
    b = D['root chord']
    c = D['leading edge tip'] - D['leading edge']
    xc, yc, A, _, Ixc, _, Iyc, _, Izc, _, Ixyc = trapezoid(h,a,b,c)

    # Find centroid location in airplane coordinates
    xc += D['leading edge']

    # Move moments of area to cg of airplane
    Dx2 = (xc - geom.CGlocation_measured[0])**2
    Dy2 = yc**2
    Ix = Ixc + Dy2*A
    Iy = Iyc + Dx2*A
    Iz = Izc + (Dy2 + Dx2)*A

    # Compute moments of inertia
    sigma = geom.balsa_density * geom.balsa_thickness
    Ix *= sigma
    Iy *= sigma
    Iz *= sigma

    # build object
    Iwing = inertias(Ixx=2*Ix, Iyy=2*Iy, Izz=2*Iz, Ixz=0, A=2*A)
    CG_est += xc * 2 * A * geom.balsa_density * geom.balsa_thickness

    # # H-tail
    D = geom.Htail

    # Estimate moments of area for a half tail at its centroid
    h = D['span']/2
    a = D['tip chord']
    b = D['root chord']
    c = D['leading edge tip'] - D['leading edge']
    xc, yc, A, _, Ixc, _, Iyc, _, Izc, _, Ixyc = trapezoid(h,a,b,c)

    # Find centroid location in airplane coordinates
    xc += D['leading edge']

    # Move moments of area to cg of airplane
    Dx2 = (xc - geom.CGlocation_measured[0])**2
    Dy2 = yc**2
    Ix = Ixc + Dy2*A
    Iy = Iyc + Dx2*A
    Iz = Izc + (Dy2 + Dx2)*A

    # Compute moments of inertia
    sigma = geom.balsa_density * geom.balsa_thickness
    Ix *= sigma
    Iy *= sigma
    Iz *= sigma

    # build object
    Ihtail = inertias(Ixx=2*Ix, Iyy=2*Iy, Izz=2*Iz, Ixz=0, A=2*A)
    CG_est += xc * 2 * A * geom.balsa_density * geom.balsa_thickness

    # # Vtail
    D = geom.Vtail

    # Estimate moments of area for the tail at its centroid
    h = D['span']
    a = D['tip chord']
    b = D['root chord']
    c = D['leading edge tip'] - D['leading edge']
    xc, zc, A, _, Ixc, _, Izc, _, Iyc, _, Ixzc = trapezoid(h,a,b,c)

    # Find centroid location in airplane coordinates
    xc += D['leading edge']

    # Move moments of area to cg of airplane
    Dx = (xc - geom.CGlocation_measured[0])
    Dz = zc
    Ix = Ixc + Dz**2*A
    Iy = Iyc + Dx**2*A
    Iz = Izc + (Dz**2 + Dx**2)*A
    Ixz = Ixzc + Dz*Dx*A

    # Compute moments of inertia
    sigma = geom.balsa_density * geom.balsa_thickness
    Ix *= sigma
    Iy *= sigma
    Iz *= sigma
    Ixz *= sigma

    # build object
    Ivtail = inertias(Ixx=Ix, Iyy=Iy, Izz=Iz, Ixz=Ixz, A=A)
    CG_est += xc * A * geom.balsa_density * geom.balsa_thickness

    # # Fuselage

    D = geom.Fuselage
    # At centroid
    I = geom.carbon_linear_density * D['length']**3 / 12

    # Move to CG location
    m = geom.carbon_linear_density*D['length']
    I += m * (D['length']/2 - geom.CGlocation_measured[0])**2

    # Add nose weight
    I += D['nose weight'] * geom.CGlocation_measured[0]**2

    # build object
    Ifuse = inertias(Ixx=0, Iyy=I, Izz=I, Ixz=0, A=0)
    CG_est += D['length']/2 * D['length'] * geom.carbon_linear_density

    M = geom.Fuselage['nose weight'] + \
        geom.Fuselage['length']*geom.carbon_linear_density + \
        (Iwing.A + Ihtail.A + Ivtail.A)*geom.balsa_thickness*geom.balsa_density

    # # Inertia matrix

    Ixx = Iwing.Ixx + Ihtail.Ixx + Ivtail.Ixx + Ifuse.Ixx #+ Iglue.Ixx + Imarkers.Ixx
    Iyy = Iwing.Iyy + Ihtail.Iyy + Ivtail.Iyy + Ifuse.Iyy #+ Iglue.Iyy + Imarkers.Iyy
    Izz = Iwing.Izz + Ihtail.Izz + Ivtail.Izz + Ifuse.Izz #+ Iglue.Izz + Imarkers.Izz
    Ixz = Iwing.Ixz + Ihtail.Ixz + Ivtail.Ixz + Ifuse.Ixz #+ Iglue.Ixz + Imarkers.Ixz
    
    return Ixx, Iyy, Izz, Ixz, M, CG_est


def run_inertia_estimation(aircraft_file):
    # Load 
    geom = AircraftGeom(aircraft_file)     

    # Structure weights 
    Ixx, Iyy, Izz, Ixz, M, CG_est = structure_inertia(geom)
    

    # Additional elements
    d_el = geom.extra_mass_elements
    CGloc = np.array(geom.CGlocation_measured)
    for element in d_el:
        print(f"Adding mass {element} to the inertia estimation")
        delta = np.array(d_el[element]['location_in_nose_frame'])
        delta -= CGloc # Remove cg location
        Ixx += (delta[1] + delta[2])**2 * d_el[element]['mass']
        Iyy += (delta[0] + delta[2])**2 * d_el[element]['mass']
        Izz += (delta[0] + delta[1])**2 * d_el[element]['mass']
        Ixz += delta[0]* delta[2] * d_el[element]['mass']
        M += d_el[element]['mass']
        CG_est += d_el[element]['mass'] * d_el[element]['location_in_nose_frame'][0]
    
    # Save updated inertia matrix
    geom.mass_estimated = M
    geom.inertia_estimated = [[Ixx,0, -Ixz], [0, Iyy,0],[-Ixz,0,Izz]]
    geom.CGlocation_estimated = [CG_est/M, 0.0, 0.0]
    geom.dump()


@click.command()
@click.option('--aircraft_file', default='aircraft.json', type=str)
def run(aircraft_file):
    run_inertia_estimation(aircraft_file)

if __name__ == '__main__':
    run()

