const express=require("express");
const body_parser=require("body-parser");
const axios=require("axios");
const bot = require("./model.js");
const tf = require('@tensorflow/tfjs');
require('dotenv').config();

const app=express().use(body_parser.json());

const token=process.env.TOKEN;
const mytoken=process.env.MYTOKEN;
// 
app.listen(process.env.PORT,()=>{
    console.log("webhook is listening");

});

//to verify the callback url from dashboard side - cloud api side
app.get("/webhook",(req,res)=>{
   let mode=req.query["hub.mode"];
   let challange=req.query["hub.challenge"];
   let token=req.query["hub.verify_token"];


    if(mode && token){

        if(mode==="subscribe" && token===mytoken){
            res.status(200).send(challange);
        }else{
            res.status(403);
        }

    }

});

app.post("/webhook",(req,res)=>{ //i want some 

    let body_param=req.body;

    if(body_param.object){
        console.log("inside body param");
        if(body_param.entry && 
            body_param.entry[0].changes && 
            body_param.entry[0].changes[0].value.messages && 
            body_param.entry[0].changes[0].value.messages[0]  
            ){
               let phon_no_id=body_param.entry[0].changes[0].value.metadata.phone_number_id;
               let from = body_param.entry[0].changes[0].value.messages[0].from; 
               let msg_body = body_param.entry[0].changes[0].value.messages[0].text.body;
               msg_body = msg_body.toString();
               axios({
                method:"POST",
                url:"https://molyhost.ga/predict",
                data:{
                    "question":msg_body
                },
                headers:{
                    "Content-Type":"application/json"
                }
               }).then((response)=>{
                     let result = response.data['prediction'];
                     result = result.toString();
                     axios({
                        method:"POST",
                        url:"https://graph.facebook.com/v13.0/"+phon_no_id+"/messages?access_token="+token,
                        data:{
                            messaging_product:"whatsapp",
                            to:from,
                            text:{
                                body:result
                            }
                        },
                        headers:{
                            "Content-Type":"application/json"
                        }
     
                    });
                }).catch((err)=>{
                    console.log(err);
                });
               
               res.sendStatus(200);
            }else{
                res.sendStatus(404);
            }

    }

});

app.get("/",(req,res)=>{
    res.status(200).send("hello this is webhook setup");
});