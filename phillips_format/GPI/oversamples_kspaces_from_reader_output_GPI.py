import numpy as np
import gpi

def DC_offset_correction(data,noise):
    corrected_data = np.zeros(data.shape,dtype=data.dtype)
    for i in range(noise.shape[0]):
        offset_value = np.mean(noise[i,:])
        corrected_data[i] = data[i]-offset_value
    return corrected_data

def oversampled_kspace_from_scan(data, header):
    # get the line positions
    yline_positions = []
    # ctrl_values = {}
    prep_phase = True

    for i, control_val in enumerate(header['lab']['control']):
        # if control_val not in ctrl_values:
        #     ctrl_values[control_val] = 0
        # ctrl_values[control_val] += 1
        if prep_phase:
            if control_val==b'CTRL_END_PREP_PHASE':
                prep_phase = False
            continue
        is_normal_data = control_val==b'CTRL_NORMAL_DATA'
        linepos = header['lab']['e1_profile_nr'][i]
        if is_normal_data and (linepos not in yline_positions):
            yline_positions.append(linepos)

    #prepare output arrays
    outshape = [s for s in data.shape]
    outshape[-1] = int(int(header['sin']['recon_resolutions'][0][0]) * float(header['sin']['oversample_factors'][0][0]))
    outshape[-2] = int(int(header['sin']['recon_resolutions'][0][1]) * float(header['sin']['oversample_factors'][0][1]))
    oversampled_kspace = np.zeros(outshape,dtype = data.dtype)

    #get the position offset at scan time
    center_oversampled_y = int(outshape[-2]/2)
    center_oversampled_x = int(outshape[-1]/2)
    offset_y = center_oversampled_y  - int(int(header['sin']['nr_e1_profiles'])/2)
    start_x = center_oversampled_x  + int(header['sin']['min_encoding_numbers'][0][0])
    end_x = center_oversampled_x  + int(header['sin']['max_encoding_numbers'][0][0]) + 1

    for i in range(len(yline_positions)):#instead of data.shape[-2]
        oversampled_kspace[:,:,:,yline_positions[i]+offset_y,start_x:end_x] = data[:,:,:,i,:]

    return oversampled_kspace

class ExternalNode(gpi.NodeAPI):
    """This is a GPI node template.

    You should replace this docstring with accurate and thorough documentation.
    INPUT:
        in - a numpy array
    OUTPUT:
        out - a numpy array
    WIDGETS:
        foo - an integer
    """

    # initialize the UI - add widgets and input/output ports
    def initUI(self):
        # Widgets
        # self.addWidget('SpinBox', 'foo', val=10, min=0, max=100)
        # self.addWidget('DoubleSpinBox', 'bar', val=10, min=0, max=100)
        self.addWidget('PushButton', 'DcOffsetCorrection', toggle=True)
        # self.addWidget('ExclusivePushButtons', 'qux',
        #              buttons=['Antoine', 'Colby', 'Trotter', 'Adair'], val=1)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64, ndim=5)
        self.addInPort('noise', 'NPYarray', dtype=np.complex64, ndim=2)
        self.addInPort('header', 'DICT')
        self.addOutPort('oversampled_kspaces', 'NPYarray', dtype=np.complex64, ndim=5)


    # validate the data - runs immediately before compute
    # your last chance to show/hide/edit widgets
    # return 1 if the data is not valid - compute will not run
    # return 0 if the data is valid - compute will run
    def validate(self):
        in_data = self.getData('data')
        in_header = self.getData('header')
        in_noise = self.getData('noise')

        # TODO: make sure the input data is valid
        # [your code here]

        return 0

    # process the input data, send it to the output port
    # return 1 if the computation failed
    # return 0 if the computation was successful 
    def compute(self):
        data = self.getData('data')
        header = self.getData('header')
        noise = self.getData('noise')
        if self.getVal('DcOffsetCorrection'):
            data = DC_offset_correction(data,noise)
        oversampled_kspace = oversampled_kspace_from_scan(data,header)
        self.setData('oversampled_kspaces', oversampled_kspace)

        return 0

if __name__=='__main__':

    import pickle
    import numpy as np
    data = np.load(r'/home/touquet/MRIData_tmp/20210127_162451_QFLOW_Ao__data.npy')
    header = pickle.load(open(r'/home/touquet/MRIData_tmp/20210127_162451_QFLOW_Ao__header.pickle','rb'))
    noise = np.load(r'/home/touquet/MRIData_tmp/20210127_162451_QFLOW_Ao__noise.npy')
    data = DC_offset_correction(data,noise)
    test = oversampled_kspace_from_scan(data,header)
    import pdb;pdb.set_trace()