import analytics as Analytics
from consolemenu import SelectionMenu
import prediction

TABLES_NAMES = [
    'Plot graphs',
    'Is the news fake?'
    ]


def main():
    menu = SelectionMenu(
        TABLES_NAMES, 
        title="What do you want to do?")
    menu.show()
    index = menu.selected_option

    if index < len(TABLES_NAMES):  
        show_plots_menu(index)
    else:
        print('Bye!')


def getInput(msg):
    print(msg)
    return input()


def show_plots_menu(tableIndex):
    options_plot = ['pie',  'subject countplot', 'worldcloud', 'time series']
    
    if tableIndex == 0:
        selectionMenu = SelectionMenu(
            options_plot,
            f'Choose plot type: ', 
            exit_option_text='Back')
        selectionMenu.show()
        index = selectionMenu.selected_option
        try:
            if (index == 0):
                Analytics.plot_pie()
            else:
                show_plots_fake_real_menu(index)
        except IndexError:
            main()
    elif tableIndex == 1:
        show_news_testing_menu()
        getInput('')
    
    main()


def show_news_testing_menu():
    options = ['Test from manual', 'Enter news']
    selectionMenu = SelectionMenu(options, exit_option_text='Back')
    selectionMenu.show()
    index = selectionMenu.selected_option
    if index == 0:
        prediction.generate_manual_testing()
        getInput('')
    if index == 1:
        news = getInput('Enter the news text: ')
        prediction.manual_testing(news)
        getInput('')
        

def show_plots_fake_real_menu(tableIndex):
    functions = [
        Analytics.plot_pie, 
        Analytics.plot_subject_countplot,
        Analytics.plot_worldcloud,
        Analytics.plot_time_series
        ]

    options = ['Fake', 'Real']

    if tableIndex == 1:
        options.append('Both')

    selectionMenu = SelectionMenu(
        options,
        exit_option_text='Back')
    selectionMenu.show()
    index = selectionMenu.selected_option

    if (index == 0):
        functions[tableIndex](real=False)
    elif (index == 1):
        functions[tableIndex](real=True)
    elif (tableIndex == 1 and index == 2):
        functions[tableIndex](both=True)


if __name__ == '__main__':
    main()
    # client.close()


