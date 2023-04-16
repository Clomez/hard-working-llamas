TIME_TO_SCROLL = 10 // in sec

const delay = ms => new Promise(res => setTimeout(res, ms));

var scroll = setInterval(function(){ window.scrollBy(0, 1000)}, 1000);

const iterate_elements = () => {
    console.log("-- Start iterator --");

    watch = "watch?"

    const elements = document.getElementsByTagName("a")
    const elements2 = document.getElementsByTagName("link")
    const elements3 = document.querySelectorAll('[href]')

    const urls = []
    const links = [...elements, ...elements2, ...elements3]
    const URLElements = []

    // Find elements w links from document
    for (let item of links) {
        if (item.href && item.href.includes(watch)) {
            URLElements.push(item)
        }
        if (item.v && item.v.includes(watch)) {
            URLElements.push(item)
        }
    }

    // Find elements w href / v attribute
    for (let i of links) {
        if (i.href && i.href.includes(watch)) {
            urls.push(i.href)
        }
        if (i.v && i.v.includes(watch)) {
            urls.push(i.v)
        }
    }

    const arr = urls.map((i => {
        return "'" + i + "',"
    }))

    // unique strings
    uniq = [...new Set(arr)].forEach((i => console.log(i)))
    console.log("-- Done --");

}

const delayedScroll = async () => {
    await delay(TIME_TO_SCROLL * 1000);
    window.clearInterval(scroll);
    iterate_elements()

};

delayedScroll()