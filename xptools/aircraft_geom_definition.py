from recordtype import recordtype
from collections import OrderedDict
import json

'''
**Important reference frame consideration: 
where geometry is concerned, we use a reference frame centered on 
the nose, with x pointing backwards, y twds the right wing, and z up.**

The estimated moments of inertia Ix, Iy, Iz, Ixz 
estimated by xptools.inertia_estimation.mass_cg_estimation are in usual body frame. 
Thankfully all the sign flips cancel out when computing these moments.'''


DEFAULT_AIRCRAFT_GEOM = {
    "Fuselage": {
        "length": 0.0,
        "nose weight": 0.0
    },
    "mass_measured": 0.0,
    "CGlocation_measured": 0.,
    "mass_estimated": 0.,
    "CGlocation_estimated": 0.,
    "pivot_point_location": [0.,0.,0.],# In the geometric frame 
    # (centered in nose, x-back, z-up)
    "Wing": {
        "span": 0.0,
        "root chord": 0.0,
        "tip chord": 0.0,
        "leading edge": 0.0,
        "leading edge tip": 0.0,
        "incidence": 0.0
    },
    "Htail": {
        "span": 0.0,
        "root chord": 0.0,
        "tip chord": 0.0,
        "leading edge": 0.0,
        "leading edge tip": 0.0,
        "incidence": 0.0
    },
    "Vtail": {
        "span": 0.0,
        "root chord": 0.0,
        "tip chord": 0.0,
        "leading edge": 0.0,
        "leading edge tip": 0.0,
    },
    "balsa_density": 0.0,
    "balsa_thickness": 0.0,
    "carbon_linear_density": 0.0,

    # Extra point-mass elements to evaluate when estimating inertia:
    "extra_mass_elements": {
        "default": {
            "location_in_nose_frame": [
    					0.0,
    					0.0,
    					0.0
            ],
            "mass": 0.0
        }
    },
    "inertia_estimated": [
        [
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0
        ]
    ],
    "mass_measurements":{
    		"distance_between_scales": None,
    		"front_scale_to_nose": None,
    		"front_scale": None,
    		"back_scale": None
    	}
    }


AircraftGeom_ = recordtype('AircraftGeom_', DEFAULT_AIRCRAFT_GEOM)

class AircraftGeom(AircraftGeom_):
    def __init__(self, aircraft_dict):
        if type(aircraft_dict) == str:
            self.filename = aircraft_dict
            with open(aircraft_dict, 'r') as f:
                aircraft_dict = json.load(f)
        else:
            self.filename = None
        super().__init__(**aircraft_dict)
    
    def dump(self, filename=None):
        if filename is None:
            assert (self.filename is not None); "Provide filename to save AircraftGeom as json"
            filename = self.filename
        with open(filename, 'w') as f:
            json.dump(self._asdict(), f, indent=4)