import java.io.File;
import java.io.IOException;
import java.util.*;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaDeltaUpdater;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DaNetwork {

    //The absolute path of the folder containing MNIST training and testing subfolders
    private static final String MNIST_DATASET_ROOT_FOLDER = "C:/Users/lyubi/OneDrive/Documents/MY_MINST/MY_MINSTY/mnist_png/";
    //Height and widht in pixel of each image
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    //The total number of images into the training and testing set
    private static final int N_SAMPLES_TRAINING = 60000;
    private static final int N_SAMPLES_TESTING = 10000;
    //The number of possible outcomes of the network for each input,
    //correspondent to the 0..9 digit classification
    private static final int N_OUTCOMES = 10;

    private static final Logger log = LoggerFactory.getLogger(DaNetwork.class);

    public static void main(String[] args) throws IOException {

        long t0 = System.currentTimeMillis();
        DataSetIterator dsi = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);

        int nEpochs = 8; // Number of training epochs

        log.info("Build model....");
        int channels = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005) // ridge regression value
                .updater(new AdaDelta())
                .weightInit(WeightInit.RELU)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels )
                        .stride(1, 1)
                        .nOut(35)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(40)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                        .nOut(600)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SOFTMAX)
                        .build())
                // InputType.convolutional for normal image
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 100 iteration
        model.setListeners(new ScoreIterationListener(1));
        log.info("Train model....");
        model.fit(dsi, nEpochs);

        DataSetIterator testDsi = getDataSetIterator( MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(testDsi);
        log.info(eval.stats());

        long t1 = System.currentTimeMillis();
        double t = (double)(t1 - t0) / 1000.0;
        log.info("\n\nTotal time: "+t+" seconds");
    }

    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {

        File folder = new File(folderPath);
        File[] digitFolders = folder.listFiles();

        NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);

        INDArray input = Nd4j.create(new int[]{ nSamples, HEIGHT*WIDTH });
        INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });

        int n = 0;
        //scan all 0..9 digit subfolders
        for (File digitFolder : digitFolders) {
            //take note of the digit in processing, since it will be used as a label
            int labelDigit = Integer.parseInt(digitFolder.getName());
            //scan all the images of the digit in processing
            File[] imageFiles = digitFolder.listFiles();
            for (File imageFile : imageFiles) {
                //read the image as a one dimensional array of 0..255 values
                INDArray img = nil.asRowVector(imageFile);
                //scale the 0..255 integer values into a 0..1 floating range
                //Note that the transform() method returns void, since it updates its input array
                scaler.transform(img);
                //copy the img array into the input matrix, in the next row
                input.putRow( n, img );
                //in the same row of the output matrix, fire (set to 1 value) the column correspondent to the label
                output.put( n, labelDigit, 1.0 );
                //row counter increment
                n++;
            }
        }

        //Join input and output matrixes into a dataset
        DataSet dataSet = new DataSet( input, output );
        //Convert the dataset into a list
        List<DataSet> listDataSet = dataSet.asList();
        //Shuffle its content randomly
        Collections.shuffle( listDataSet, new Random(System.currentTimeMillis()) );
        //Set a batch size
        int batchSize = 1000;
        //Build and return a dataset iterator that the network can use
        DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, batchSize );
        return dsi;
    }
}
