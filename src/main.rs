mod perceptrone;
use perceptrone::*;
use std::fs::*;
use std::io::*;

const DIMS : usize = 4;

#[derive(PartialEq)]
enum IrisType {
    Setosa,
    Versicolor,
    Virginica
}

struct Record {
    values : [f32; DIMS],
    result : IrisType,
}

const EPOCHS : usize = 50;
const LEARN_SPEED_BASE : f32 = 0.0001;
const LEARN_SPEED_MUL : f32 = 0.98;

fn main() {
    println!("Hello, world!");

    let mut perceptrone : Perceptrone<DIMS> = Perceptrone::new::<3>(Some(0.25));
    perceptrone.print();

    let filename = "src/data/iris.data";
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut records : Vec<Record> = Vec::new();
    
    for (index, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let tokens : Vec<&str> = line.split(",").collect();

        if tokens.len() < (DIMS + 1){
            println!("Ignored line {}.", index);
            break;
        }

        let mut values = [0.0; DIMS];
        for d in 0..DIMS {
            let val = match tokens[d].parse::<f32>() {
                Ok(val) => val,
                Err(_e) => panic!("Invalid input.")
            };
            values[d] = val;
        }
        let result : IrisType = match tokens.get(DIMS).unwrap() {
            &"Iris-setosa" => IrisType::Setosa,
            &"Iris-versicolor" => IrisType::Versicolor,
            &"Iris-virginica" => IrisType::Virginica,
            _ => panic!("Invalid input.")
        };

        let record = Record{
            values : values,
            result : result
        };
        records.push(record);
    }

    println!("--- Learning starts ---");
    println!("Data set size: {}", records.len());
    println!("Planned epochs: {}", EPOCHS);
    println!("Base learning speed: {}", LEARN_SPEED_BASE);

    let mut current_learn_speed = LEARN_SPEED_BASE;
    let mut previous_accuracy : f32 = 1.0 / DIMS as f32;
    for i in 0..EPOCHS {
        let mut valid_results = 0;

        for record in records.iter() {
            let valid_result = perceptrone.learn(
                record.values, 
                current_learn_speed, 
                record.result == IrisType::Virginica
            );

            if valid_result {
                valid_results += 1;
            }
        }

        let accuracy : f32 = (valid_results as f32) / (records.len() as f32);
        let accuracy_delta = accuracy - previous_accuracy;
        println!("Epoch {}. Accuracy={:.2}% Diff={:.2}% LeanSpeed={:.6}", 
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
