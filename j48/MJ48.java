package j48;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;



public class MJ48 extends Classifier{
	
	j48.J48[] mj48 = new j48.J48[5];
	weka.classifiers.trees.J48[] wekaj48 = new weka.classifiers.trees.J48[5];
	double[] accuracy = new double[5];
	double[] accuracy1 = new double[5];
	static Instances[] result = new Instances[20];
	public static void main(String[] args) {
		try {

	//		j48.J48 temp = new j48.J48();
	//		weka.classifiers.trees.J48 wekaj48 = new weka.classifiers.trees.J48();
			// read data
			System.out.println("Loading Data...");
			Instances rawData = DataSource.read("data/covtype.arff");
	//		rawData.setClassIndex(rawData.numAttributes() - 1);

	//		System.out.println("Resample...");
	//		result = resample(rawData);
			
			MJ48 temp = new MJ48();
			
			loadData();
			
			temp.buildClassifier(rawData);
			
			System.out.println("Evaluating...");
			for (int i = 0; i<5; i++){
				Evaluation eval = new Evaluation(result[i*2]);
				eval.evaluateModel(temp, result[i*2+1]);
				System.out.println("Accuracy: "+(1-eval.errorRate()));
			}
	/*		System.out.println("Resample...");
			result = resample(rawData); */

			// Instances fData = filterData(trainData);

		/*	for (int i = 0; i < 10; i++) {
				System.out.println("n:"+i);
				temp.buildClassifier(result[i*2]);

				Evaluation eval = new Evaluation(result[i*2]);
				eval.evaluateModel(temp, result[i*2+1]);
				System.out.println("new mj48 :"+(1 - eval.errorRate()));
				
				wekaj48.buildClassifier(result[i*2]);
				
				eval.evaluateModel(wekaj48, result[i*2+1]);
				System.out.println("orig mj48:"+(1 - eval.errorRate()));
			}*/
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	static void loadData() throws Exception{
		
		for (int i =0; i<10; i++){
		//	System.out.println("data/test"+i*2+".arff");
			result[i*2] = DataSource.read("data/test"+i+".arff");
			result[i*2].setClassIndex(result[i*2].numAttributes()-1);
			result[i*2+1] = DataSource.read("data/train"+i+".arff");
			result[i*2+1].setClassIndex(result[i*2+1].numAttributes()-1);
		}
	}
	static Instances filterData(Instances td) {
		Instances temp = new Instances(td);

		for (int i = 0; i < 10; i++) {

			double[] currAttr = temp.attributeToDoubleArray(i);

			AttributeStats aStats = temp.attributeStats(i);

			for (int j = 0; j < currAttr.length; j++)
				System.out.println(currAttr[j]);
		}

		return td;

	}

	// resample of covtype.data
	//
	private static Instances[] resample(Instances data) throws Exception {

		// first randomize data
		Random rand = new Random((int) System.currentTimeMillis());
		data.randomize(rand);

		Instances[] resampleData = new Instances[20];

		Instances largeTrain = new Instances(data, 0, data.numInstances() / 2);
		Instances largeTest = new Instances(data, data.numInstances() / 2,
				data.numInstances() / 2);

		for (int i = 0; i < 10; i++) {

			// supervised resample
			weka.filters.supervised.instance.Resample sr = new weka.filters.supervised.instance.Resample();
			sr.setNoReplacement(true);
			sr.setSampleSizePercent(10); // 10% 29000
			sr.setRandomSeed((int) System.currentTimeMillis());
			// set sample input
			sr.setInputFormat(largeTrain);

			resampleData[i * 2] = Filter.useFilter(largeTrain, sr);
			System.out.println(resampleData[i * 2].numInstances());

			sr.setInputFormat(largeTest);
			resampleData[i * 2 + 1] = Filter.useFilter(largeTest, sr);
			System.out.println(resampleData[i * 2 + 1].numInstances());

			ArffSaver saver = new ArffSaver();
			saver.setInstances(resampleData[i * 2]);
			saver.setFile(new File("data/train" + i + ".arff"));
			saver.writeBatch();

			saver.setInstances(resampleData[i * 2 + 1]);
			saver.setFile(new File("data/test" + i + ".arff"));
			saver.writeBatch();
		}

		return resampleData;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
		System.out.println("Building...");
		
		double total = 0.0;
		for (int i = 0; i<5; i++){
			
			mj48[i] = new j48.J48();
			mj48[i].buildClassifier(result[i*2]);
			
//			wekaj48[i] = new weka.classifiers.trees.J48();
//			wekaj48[i].buildClassifier(result[i*2]);
			System.out.println("Number of nodes : "+mj48[i].m_root.numNodes());
			System.out.println("Number of leaves: "+mj48[i].m_root.numLeaves());
//			System.out.println("Time: "+mj48[i].m_root);
			
			accuracy1[i] = mj48[i].m_root.numNodes();
			total+=accuracy1[i];
			
			Evaluation tEval = new Evaluation(result[i*2]);
			
			tEval.evaluateModel(mj48[i],result[i*2+1]);
			accuracy[i] = (1-tEval.errorRate());
			System.out.println("no "+i+" :"+accuracy[i]);
		//	wekaj48[i] = new weka.classifiers.trees.J48();
		//	wekaj48[i].buildClassifier(result[i*2]);
		}
		
		total/=5;
		for (int i = 0; i<5; i++){
			accuracy1[i]/=total;
		}
		
	}
	public double classifyInstance(Instance inst) throws Exception{
		double classId = 0;
		
		double[] vote = new double[inst.numClasses()];
		
		for (int i = 0; i<5; i++){
			vote[(int)mj48[i].classifyInstance(inst)]+= accuracy[i];
		//	vote[(int)wekaj48[i].classifyInstance(inst)]+= accuracy[i];
		}
		
		double max = -100;
		for (int i = 0; i<inst.numClasses(); i++)
			if (vote[i]>max){
				max =  vote[i];
				classId = i;
			}
		return classId;
	}
}
