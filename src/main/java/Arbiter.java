
import org.bytedeco.opencv.presets.opencv_core;
import org.deeplearning4j.arbiter.conf.updater.AdaGradSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.ConvolutionLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.layers.SubsamplingLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.data.MnistDataProvider;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.cpu.nativecpu.CpuAffinityManager;


import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static java.sql.DriverManager.println;

public class Arbiter {

    public static void main(String[] args) throws IOException {

        IntegerParameterSpace firstConvLayerSize  = new IntegerParameterSpace(10,40);
        IntegerParameterSpace secondConvLayerSize  = new IntegerParameterSpace(10,60);
        IntegerParameterSpace denseLayerSize  = new IntegerParameterSpace(300,1200);

        System.out.println("check 1");

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .l2(0.0005)
                .seed(123)
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .addLayer(new ConvolutionLayerSpace.Builder()
                        .kernelSize(3,3)
                        .nOut(firstConvLayerSize)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .addLayer(new SubsamplingLayerSpace.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
                .addLayer(new ConvolutionLayerSpace.Builder()
                        .kernelSize(3,3)
                        .nOut(secondConvLayerSize)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .addLayer(new SubsamplingLayerSpace.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                        .nOut(denseLayerSize)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .addLayer(new OutputLayerSpace.Builder()
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        RandomSearchGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);

        int nTrainEpochs = 1;
        int batchSize = 64;

        MnistDataProvider dataProvider = new MnistDataProvider(nTrainEpochs, batchSize);

        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Metric.ACCURACY);

        MaxTimeCondition terminationConditions = new MaxTimeCondition(30, TimeUnit.MINUTES);

        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if(f.exists()) f.delete();
        f.mkdir();
        FileModelSaver modelSaver = new FileModelSaver(baseSaveDirectory);

        System.out.println("check 2");

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        LocalOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());

        //Start the hyperparameter optimization

        System.out.println("check 3");

        runner.execute();

        System.out.println("check 4");

        String s = "Best score: " + runner.bestScore() + "\n" + "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" + "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference>	 allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();


        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());
    }


}
