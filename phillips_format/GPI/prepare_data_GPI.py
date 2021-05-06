import numpy as np
import gpi

# This is a template node, with stubs for initUI() (input/output ports,
# widgets), validate(), and compute().
# Documentation for the node API can be found online:
# http://docs.gpilab.com/NodeAPI


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

def centered_array_into_shape(array,shape):
    """Creates a new array of the given shape and fills it 
    respecting centering using values in given array.
    If a dimension in the shape is bigger than the array
    zerofilling is used, if the dimension is smaller 
    cropping is used. Only considers the last two dims of shape.
    """
    x_margin = int(array.shape[-1]/2) - int(shape[-1]/2)
    y_margin = int(array.shape[-2]/2) - int(shape[-2]/2)
    if x_margin>0:
        xinds = [0,-1,x_margin,-1*(x_margin+1)]
    else:
        x_margin *= -1
        xinds = [x_margin,-1*(x_margin+1),0,-1]
    if y_margin>0:
        yinds = [0,-1,y_margin,-1*(y_margin+1)]
    else:
        y_margin *= -1
        yinds = [y_margin,-1*(y_margin+1),0,-1]
    newshape = [s for s in array.shape]
    newshape[-2:] = shape[-2:]
    output = np.zeros(newshape,dtype=array.dtype)
    output[...,yinds[0]:yinds[1],xinds[0]:xinds[1]] = array[...,yinds[2]:yinds[3],xinds[2]:xinds[3]]
    return output

def reconprep_images(data, oversampled_images,header):
    #first, roll in the phase dimension
    center_oversampled = int(data.shape[-2]/2)
    oversample_offset = int(header['sin']['oversample_offsets'][0][1]) if 'oversample_offsets' in header['sin'] else 0
    displacement = center_oversampled + oversample_offset
    oversampled_images = np.roll(oversampled_images,331,axis=-2)

    #then remove oversampling + crop to output res like philips
    recon_shapes = (int(header['sin']['recon_resolutions'][0][1]), int(header['sin']['recon_resolutions'][0][0]))
    recon_images = centered_array_into_shape(oversampled_images,recon_shapes)
    output_shapes = (int(header['sin']['output_resolutions'][0][1]), int(header['sin']['output_resolutions'][0][0]))
    output_images = centered_array_into_shape(recon_images,output_shapes)
    return output_images, recon_images


def compute_from_args(data, header):

    oversampled_kspaces = oversampled_kspace_from_scan(data, header)
    oversampled_images = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(oversampled_kspaces,axes=(-2,-1)),axes=(-2,-1)),axes=(-2,-1))
    output_images, recon_images = reconprep_images(data, oversampled_images,header)
    return output_images, recon_images

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
        self.addWidget('PushButton', 'recon_format', toggle=True)
        # self.addWidget('ExclusivePushButtons', 'qux',
        #              buttons=['Antoine', 'Colby', 'Trotter', 'Adair'], val=1)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64, ndim=5)
        self.addInPort('header', 'DICT')
        self.addOutPort('output_images', 'NPYarray', dtype=np.complex128, ndim=5)


    # validate the data - runs immediately before compute
    # your last chance to show/hide/edit widgets
    # return 1 if the data is not valid - compute will not run
    # return 0 if the data is valid - compute will run
    def validate(self):
        in_data = self.getData('data')
        in_header = self.getData('header')

        # TODO: make sure the input data is valid
        # [your code here]

        return 0

    # process the input data, send it to the output port
    # return 1 if the computation failed
    # return 0 if the computation was successful 
    def compute(self):
        data = self.getData('data')
        header = self.getData('header')

        output_images, recon_images = compute_from_args(data,header)
        # TODO: process the data
        # [your code here]
        recon_format = self.getVal('recon_format')
        if recon_format:
            out = recon_images
        else:
            out = output_images

        self.setData('output_images', out)

        return 0

if __name__=='__main__':

    import pickle
    import numpy as np
    data = np.load(r'C:/Users/touquet/Desktop/for gpi/data.npy')
    header = pickle.load(open(r'C:\Users\touquet\Desktop\for gpi\header.pickle','rb'))
    test = compute_from_args(data,header)