import numpy as np
import gpi

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

def reconprep_images(oversampled_images,header):
    #first, roll in the phase dimension
    center_oversampled = int(oversampled_images.shape[-2]/2)
    oversample_offset = int(header['sin']['oversample_offsets'][0][1]) if 'oversample_offsets' in header['sin'] else 0
    displacement = center_oversampled + oversample_offset
    oversampled_images = np.roll(oversampled_images,displacement,axis=-2)

    #then remove oversampling + crop to output res like philips
    recon_shapes = (int(header['sin']['recon_resolutions'][0][1]), int(header['sin']['recon_resolutions'][0][0]))
    recon_images = centered_array_into_shape(oversampled_images,recon_shapes)
    output_shapes = (int(header['sin']['output_resolutions'][0][1]), int(header['sin']['output_resolutions'][0][0]))
    output_images = centered_array_into_shape(recon_images,output_shapes)
    return output_images, recon_images, oversampled_images

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
        self.addInPort('images', 'NPYarray', dtype=np.complex64, ndim=5)
        self.addInPort('header', 'DICT')
        self.addOutPort('output_images', 'NPYarray', dtype=np.complex64, ndim=5)
        self.addOutPort('oversampled_images', 'NPYarray', dtype=np.complex64, ndim=5)


    # validate the data - runs immediately before compute
    # your last chance to show/hide/edit widgets
    # return 1 if the data is not valid - compute will not run
    # return 0 if the data is valid - compute will run
    def validate(self):
        in_data = self.getData('images')
        in_header = self.getData('header')

        # TODO: make sure the input data is valid
        # [your code here]

        return 0

    # process the input data, send it to the output port
    # return 1 if the computation failed
    # return 0 if the computation was successful 
    def compute(self):
        data = self.getData('images')
        header = self.getData('header')
        output_images, recon_images, oversampled_images = reconprep_images(data,header)
        # TODO: process the data
        # [your code here]
        recon_format = self.getVal('recon_format')
        if recon_format:
            out = recon_images
        else:
            out = output_images

        self.setData('output_images', out)
        self.setData('oversampled_images', oversampled_images)

        return 0

if __name__=='__main__':

    import pickle
    import numpy as np
    # from .oversamples_kspaces_from_reader_output_GPI import DC_offset_correction, oversampled_kspace_from_scan
    data = np.load(r'/home/touquet/MRIData_tmp/20210127_162451_QFLOW_Ao__oversampled_images.npy')
    header = pickle.load(open(r'/home/touquet/MRIData_tmp/20210127_162451_QFLOW_Ao__header.pickle','rb'))
    # oversampled_kspaces = oversampled_kspace_from_scan(data,header)
    test = reconprep_images(data,header)
    import pdb;pdb.set_trace()