mod perceptrone;
use perceptrone::*;
use std::fs::*;
use std::io::*;

const IRIS_DIM : usize = 4;

#[derive(PartialEq)]
#[derive(Copy, Clone)]
enum IrisType {
    Setosa,
    Versicolor,
    Virginica
}

struct Record {
    values : [f32; IRIS_DIM],
    result : IrisType,
}


const EPOCHS : usize = 20;
const LEARN_SPEED_BASE : f32 = 0.003;
const LEARN_SPEED_MUL : f32 = 1.0;
const DATA_FILE_PATH : &str = "data/iris.data";
const SEARCHED_IRIS : IrisType = IrisType::Virginica;


fn main() {
    let mut records : Vec<Record> = Vec::new();
    {
        println!("Reading file with data: {}", DATA_FILE_PATH);

        let file = File::open(DATA_FILE_PATH).unwrap();
        let reader = BufReader::new(file);
        
        for (index, line) in reader.lines().enumerate() {
            let line = line.unwrap();
            let tokens : Vec<&str> = line.split(",").collect();

            if tokens.len() < (IRIS_DIM + 1){
                println!("Ignored line {}.", index);
                break;
            }

            let mut values = [0.0; IRIS_DIM];
            for d in 0..IRIS_DIM {
                let val = match tokens[d].parse::<f32>() {
                    Ok(val) => val,
                    Err(_e) => panic!("Invalid input.")
                };
                values[d] = val;
            }
            let result : IrisType = match tokens.get(IRIS_DIM).unwrap() {
                &"Iris-setosa" => IrisType::Setosa,
                &"Iris-versicolor" => IrisType::Versicolor,
                &"Iris-virginica" => IrisType::Virginica,
                _ => panic!("Invalid input.")
            };
            
            records.push(Record{ values, result });
        }
    }


    println!("--- Learning starts ---");
    println!("Data set size: {}", records.len());
    println!("Planned epochs: {}", EPOCHS);
    println!("Base learning speed: {}", LEARN_SPEED_BASE);
    println!("Searched iris index: {}", SEARCHED_IRIS as usize);

    let mut perceptrone : Perceptrone<IRIS_DIM> = Perceptrone::new::<3>(Some(0.25));
    perceptrone.print();

    let mut current_learn_speed = LEARN_SPEED_BASE;
    let mut previous_accuracy : f32 = 1.0 / IRIS_DIM as f32;
    for i in 0..EPOCHS {
        let mut valid_classifications = 0;

        for record in records.iter() {
            let classification_valid_during_learning = perceptrone.learn(
                record.values, current_learn_speed, record.result == SEARCHED_IRIS);
            
            if classification_valid_during_learning {
                valid_classifications += 1;
            }
        }

        let accuracy : f32 = (valid_classifications as f32) / (records.len() as f32);
        let accuracy_delta = accuracy - previous_accuracy;
        println!("Epoch {}. Accuracy={:.2}% Diff={:.2}% LearnSpeed={:.3}", 
            i,
            accuracy * 100.0,
            accuracy_delta * 100.0,
            current_learn_speed
        );
        perceptrone.print();

        current_learn_speed *= LEARN_SPEED_MUL;
        previous_accuracy = accuracy;
    }
}