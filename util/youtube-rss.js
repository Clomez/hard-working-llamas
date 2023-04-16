
// Some reason, only works some of the time..

const delay = ms => new Promise(res => setTimeout(res, ms));

var scroll = setInterval(function(){ window.scrollBy(0, 1000)}, 1000);

const RSS = "/feeds/"
const URLs = []

const elements = document.getElementsByTagName("a")
const elements2 = document.getElementsByTagName("link")
const elements3 = document.querySelectorAll('[href]')

const test = [...elements, ...elements2, ...elements3]

const iterate_elements = () => {

    watch = "watch/"

    const elements = document.getElementsByTagName("a")
    const elements2 = document.getElementsByTagName("link")
    const elements3 = document.querySelectorAll('[href]')

    const links = [...elements, ...elements2, ...elements3]

    for (let item of links) {
        if (item.href && item.href.includes(RSS)) {
            URLs.push(item)
        }
    }

    URLs.forEach((i => {
        console.log(i)
    }))
    
}

for (let item of test) {
    if (item.href && item.href.includes(RSS)) {
        URLs.push(item)
    }
}

if (URLs.length == 0) {
    delayedScroll();
} 

console.log("Printing..")
URLs.forEach((i => {
    console.log(i)
}))

const delayedScroll = async () => {

    await delay(5000);
    console.log("-- Finished --");
    window.clearInterval(scroll);
    iterate_elements()

};

delayedScroll()

