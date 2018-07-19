NeuralNetwork nn;
Reader r;
byte[] cat_data;
byte[] rainbow_data;
byte[] train_data;

float ratio_testing = 0.2;
float ratio_training = 0.8;


int total = 30000;
int no_doodles = 3;
int nr_testing  = (int)(no_doodles*total*ratio_testing);
int nr_training = (int)(no_doodles*total*ratio_training);
int nrGen = 0;

Image[] training;
Image[] testing;

void setup() {

  nn = new NeuralNetwork(784, 128, 3);
  r = new Reader();
  //r.write(nn);
  r.read(nn);
  training = new Image[nr_training];
  testing = new Image[nr_testing];
  cat_data = loadBytes("data/cat.npy");
  rainbow_data = loadBytes("data/rainbow.npy");
  train_data = loadBytes("data/train.npy");
  ///////////////////////////////////////////////0 = cat////1 = rainbow////2 = train//////////////////////
  prep_data();
  size(280, 280);
  println("This is correct: "+Test()+"%");
  //for (int i=0; i<48; i++) {
  //  EvolveGen();
  //  r.write(nn);
  //  println("This is correct: "+Test()+"%");
  //}


  background(0);
}

void draw() {
  draw_yourself();
}

boolean new_image = true;

void draw_yourself() {
  stroke(255);
  strokeWeight(16);
  if (mousePressed)
    line(pmouseX, pmouseY, mouseX, mouseY);
  if (keyPressed) {
    if (key == 'a') {
      background(0);
      new_image = true;
    }
    if (key == ENTER && new_image) {
      PImage img = get();
      float[] inputs = new float[784];
      img.resize(28, 28);
      img.loadPixels();
      for (int i=0; i<784; i++) {
        int c=img.pixels[i]; // so we don't access the array too much
        int r=(c&0x00FF0000)>>16; // red part
        int g=(c&0x0000FF00)>>8; // green part
        int b=(c&0x000000FF); // blue part
        int grey=(r+b+g)/3;
        inputs[i] = grey/255.0;
      }
      float[] outputs = nn.estimate(inputs);
      float max = max(outputs[0], max(outputs[1], outputs[2]));
      int guess = 2;
      if (outputs[0] == max) guess = 0;
      else if (outputs[1] == max) guess = 1;
      String str = "";
      switch(guess) {
      case 0:
        {
          str = "o pisica";
          break;
        }
      case 1:
        {
          str = "un curcubeu";
          break;
        }
      case 2:
        {
          str = "un tren";
          break;
        }
      }
      //textAlign(CENTER,CENTER);
      textSize(18);
      fill(100, 255, 255);
      background(155, 0, 0);
      text("Ai desenat "+str+". \nApasa \'a\' pentru a desena \ninca ceva.", 10, 30);
      new_image = false;
    }
  }
}

void prep_data() {
  for (int i=0; i<nr_training; i++) {
    training[i] = new Image((int)i/(nr_training/3));
    training[i].id = i;
  }
  for (int i=0; i<nr_testing; i++) {
    testing[i] = new Image((int)i/(nr_testing/3));
  }


  for (int i=0; i<total; i++) {
    int start = 80 + i * 784;
    if (i<nr_training/3) {
      for (int j=0; j<784; j++) {
        training[i].data[j] = (cat_data[start+j] & 0xff)/255.0;
      }
    } else {
      for (int j=0; j<784; j++) {
        testing[i-4*nr_testing/3].data[j] = (cat_data[start+j] & 0xff)/255.0;
      }
    }
  }

  for (int i=0; i<total; i++) {
    int start = 80 + i * 784;
    if (i<nr_training/3) {
      for (int j=0; j<784; j++) {
        training[i+nr_training/3].data[j] = (rainbow_data[start+j] & 0xff)/255.0;
      }
    } else {
      for (int j=0; j<784; j++) {
        testing[i-nr_testing].data[j] = (rainbow_data[start+j] & 0xff)/255.0;
      }
    }
  }

  for (int i=0; i<total; i++) {
    int start = 80 + i * 784;
    if (i<nr_training/3) {
      for (int j=0; j<784; j++) {
        training[i+2*nr_training/3].data[j] = (train_data[start+j] & 0xff)/255.0;
      }
    } else {
      for (int j=0; j<784; j++) {
        testing[i-2*nr_testing/3].data[j] = (train_data[start+j] & 0xff)/255.0;
      }
    }
  }
}

void EvolveGen() {
  training = randomize(training);
  for (int i=0; i<nr_training; i++) {
    float target[] = {0, 0, 0};
    target[training[i].label] = 1;
    nn.evolve(training[i].data, target);
  }
  nrGen++;
  println("Generation: "+nrGen);
}

float Test() {
  int counter = 0;
  for (int i=0; i<nr_testing; i++) {
    float[] output = nn.estimate(testing[i].data);
    float max = max(output[0], max(output[1], output[2]));
    float guess = 2;
    if (output[0] == max) guess = 0;
    else if (output[1] == max) guess = 1;
    if (guess == testing[i].label)
      counter++;
  }
  return counter/(nr_testing/100.0);
}

class Image {
  public int label;
  public int id;
  public float[] data = new float[784];
  Image(int a) {
    label = a;
  }
  void normalise() {
    for (int i=0; i<this.data.length; i++) {
      this.data[i] /= 255;
    }
  }
}

Image[] randomize(Image[] image) {

  Image[] img = image;
  Image aux;
  int index;
  for (int i = nr_training - 1; i > 0; i--)
  {
    index = (int)random(i + 1);
    if (index != i)
    {
      aux = img[i];
      img[i] = img[index];
      img[index] = aux;
    }
  }

  return img;
}
